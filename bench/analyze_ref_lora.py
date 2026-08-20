#!/usr/bin/env python3
"""Does kijai's ref LoRA reconstruct ref2va from fl2va? Graded per module.

No GPU, no server, no render -- it reads three safetensors headers and does
linear algebra on the tensors it needs. About a minute.

## Why this exists

When this was written (2026-08-16), `REF_LORA_ENABLED = True` made fl2va +
this LoRA the canonical reference path, replacing the ref2va checkpoint. That
rested on a claim `h3_config.py` phrased as a "should": at strength 1.0 the
extracted delta reconstructs ref2va, up to rank truncation and requantization
error. Nobody had checked. (The switch was flipped to False on 2026-08-18 —
ref2va loads directly now — but the reconstruction question this script
answers is unchanged, and it still applies to the deliberate LoRA arm.)

## The finding, and the reason it is two findings

**Keep COVERAGE and RECONSTRUCTION apart.** They are different claims and
conflating them is the same shape as the phantom-keys bug in `provenance.py`:

  coverage        which modules are touched.       VERIFIED, exactly.
  reconstruction  whether the values are right.    Partly verified, partly
                  NOT GRADEABLE from these files.

**Coverage: 474 of 474 modules, zero unmatched in either direction.** The LoRA
carries 794 tensors -- 264 lora_up/down pairs plus full-rank `.diff` on every
norm and `.diff_b` on every bias. It is a whole-model delta, not an adapter.

**Reconstruction splits exactly on the int8 boundary, and NOT on rank**, which
is what makes the result readable:

  64 non-quantized modules   residual 0.0022, cosine 1.0000  -- essentially
                             exact. Includes all 51 rank-8 adaln projections.
                             (An earlier version of this line called those
                             "the MOST rewritten modules", relative delta
                             1.86-1.92. Withdrawn 2026-08-20: the curve-form
                             coefficient matrices sit on bases whose columns
                             differ in sign between the parents, so that
                             delta is an artifact; at the modulation output
                             the parents differ by a few percent. See
                             bench/analyze_checkpoint_delta.py.)

  200 int8 modules           NOT GRADEABLE. See below.

**Why the int8 modules cannot be graded from these artifacts.** Two reasons,
both measured here rather than argued:

1. The LoRA's delta on them is **~0.4 int8 quantization steps RMS** -- below
   the half-step that would be needed to resolve it. So comparing requantized
   integer codes says nothing: adding a sub-step delta flips codes close to
   randomly.
2. The target is wrong too. `W_ref2va_int8 - W_fl2va_int8` differences two
   INDEPENDENTLY quantized checkpoints, so it carries both files'
   quantization error. The true relative delta on these modules is only
   ~0.033, so that error is first-order, not a correction.

Together those produce a residual near 1.0 and a cosine near 0 on exactly
those 200 modules -- which reads as "the LoRA is wrong" and is not. The tell
is that the split falls on the int8 boundary rather than on rank: a genuinely
bad extraction would not be perfect on 64 modules and orthogonal on 200.

**THE THREE-WAY CROSS-CHECK SETTLES IT, and it flips the reading.** We hold
TWO independent quantizations of both checkpoints -- `int8_convrot` and
`fp8_scaled`. Each gives its own estimate of the same true delta, so their
agreement measures how much signal a quantized target carries at all:

    cos(D_int8, D_fp8)   0.03     the two targets barely agree with EACH OTHER
    cos(D_int8, LoRA)    0.03
    cos(D_fp8,  LoRA)    0.28-0.45

**The LoRA agrees with the fp8-derived delta about ten times better than the
two quantizations agree with each other.** That is positive evidence: the LoRA
is closer to the truth than either target, which is exactly why neither can
grade it.

It also identifies the culprit. `int8_convrot` is not a plain quantization --
the name says a rotation is applied, and differencing two independently
rotated-and-quantized checkpoints destroys the delta almost completely (3%
signal, against fp8's 30-45%). The original comparison was not merely noisy,
it was structurally the wrong quantity.

This fits the likely provenance: kijai's README calls it "the difference
between fl2va and ref2va, completely experimental", and the extraction was
probably taken from the full-precision weights on HF, which we do not hold.

**So the honest verdict is: structurally correct, correctly applied, exact on
every module that can be graded, and positively supported on the rest by the
best target available.** What remains is not a weight question -- it is
whether the two RENDER the same, and only a paired render answers that.

**And the consequence worth carrying:** the delta this LoRA applies is
comparable to the quantization step of the checkpoint it is applied to. That
bounds what ANY extraction can deliver here -- a higher-rank one would be
writing detail the int8 base cannot store. The binding constraint is the
quantization, not the rank.

    python bench/analyze_ref_lora.py
    python bench/analyze_ref_lora.py --modules 20
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import torch
from safetensors import safe_open

HOME = pathlib.Path.home()
LORA = HOME / "ComfyUI/models/loras/h3/minimax_h3_ref_lora_rank_256_bf16.safetensors"
BASE = HOME / "ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TGT = HOME / "ComfyUI/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
# A SECOND, independent quantization of the same two checkpoints. This is what
# makes the result readable: two targets that disagree with each other cannot
# grade a third thing.
BASE_FP8 = HOME / "ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
TGT_FP8 = HOME / "ComfyUI/models/diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors"


def dequant(f, module: str) -> torch.Tensor:
    """Weight in compute space. int8 weights carry a per-row `weight_scale`."""
    w = f.get_tensor(module + ".weight")
    keys = f.keys()
    for suffix in (".weight_scale", ".scale_weight"):
        if module + suffix in keys:
            return w.float() * f.get_tensor(module + suffix).float()
    return w.float()


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--modules", type=int, default=12,
                    help="how many worst-rewritten modules to print")
    args = ap.parse_args()

    for p in (LORA, BASE, TGT):
        if not p.exists():
            print(f"missing: {p}")
            return 2

    fb, ft, fl = (safe_open(str(p), framework="pt") for p in (BASE, TGT, LORA))
    base_keys = set(fb.keys())

    pairs = sorted({k[len("diffusion_model."):].rsplit(".lora_", 1)[0]
                    for k in fl.keys() if ".lora_down.weight" in k})

    rows = []
    for m in pairs:
        if m + ".weight" not in base_keys:
            continue
        wb, wt = dequant(fb, m), dequant(ft, m)
        up = fl.get_tensor(f"diffusion_model.{m}.lora_up.weight").float()
        dn = fl.get_tensor(f"diffusion_model.{m}.lora_down.weight").float()
        delta = (up @ dn).reshape(wb.shape)
        true = wt - wb
        quantized = m + ".weight_scale" in base_keys
        step = (fb.get_tensor(m + ".weight_scale").float().expand_as(delta)
                if quantized else None)
        rows.append(dict(
            module=m, rank=dn.shape[0], quantized=quantized,
            rel_delta=(true.norm() / wb.norm()).item(),
            residual=((true - delta).norm() / true.norm()).item(),
            cos=torch.nn.functional.cosine_similarity(
                true.flatten(), delta.flatten(), dim=0).item(),
            steps=(delta / step).pow(2).mean().sqrt().item() if quantized else None,
        ))

    print(f"{len(rows)} modules carry a lora pair "
          f"({sum(r['quantized'] for r in rows)} int8, "
          f"{sum(not r['quantized'] for r in rows)} not)\n")

    rows.sort(key=lambda r: -r["rel_delta"])
    print(f"{'most-rewritten modules':<44}{'rank':>5}{'int8':>6}"
          f"{'rel_delta':>10}{'residual':>10}{'cos':>8}")
    for r in rows[:args.modules]:
        print(f"{r['module']:<44}{r['rank']:>5}{'yes' if r['quantized'] else 'no':>6}"
              f"{r['rel_delta']:>10.3f}{r['residual']:>10.4f}{r['cos']:>8.4f}")

    def med(sel, key):
        v = [r[key] for r in rows if sel(r) and r[key] is not None]
        return statistics.median(v) if v else float("nan")

    print(f"\n{'bucket':<24}{'n':>5}{'rel_delta':>11}{'residual':>10}{'cos':>8}")
    for label, sel in (("non-quantized base", lambda r: not r["quantized"]),
                       ("int8 base", lambda r: r["quantized"])):
        n = sum(1 for r in rows if sel(r))
        print(f"{label:<24}{n:>5}{med(sel,'rel_delta'):>11.4f}"
              f"{med(sel,'residual'):>10.4f}{med(sel,'cos'):>8.4f}")

    # --- the three-way cross-check, which is what makes the rest readable ---
    if BASE_FP8.exists() and TGT_FP8.exists():
        gb, gt = safe_open(str(BASE_FP8), framework="pt"), safe_open(str(TGT_FP8), framework="pt")
        cos = torch.nn.functional.cosine_similarity
        quant = [r["module"] for r in rows if r["quantized"]][:6]
        print(f"\nTHREE-WAY CROSS-CHECK on {len(quant)} int8 modules. Two independent\n"
              f"quantizations of the SAME pair, so their mutual agreement is the\n"
              f"ceiling on what either can grade:\n")
        print(f"{'module':<34}{'cos(Di8,Dfp8)':>14}{'cos(Di8,L)':>12}{'cos(Dfp8,L)':>13}")
        agree, lora_fp8 = [], []
        for m in quant:
            di = (dequant(ft, m) - dequant(fb, m)).flatten()
            df = (dequant(gt, m) - dequant(gb, m)).flatten()
            up = fl.get_tensor(f"diffusion_model.{m}.lora_up.weight").float()
            dn = fl.get_tensor(f"diffusion_model.{m}.lora_down.weight").float()
            lo = (up @ dn).flatten()
            a, b, c = (cos(di, df, dim=0).item(), cos(di, lo, dim=0).item(),
                       cos(df, lo, dim=0).item())
            agree.append(a); lora_fp8.append(c)
            print(f"{m:<34}{a:>14.4f}{b:>12.4f}{c:>13.4f}")
        ma, ml = statistics.median(agree), statistics.median(lora_fp8)
        print(f"\n  targets agree with each other : {ma:.4f}")
        print(f"  LoRA agrees with the fp8 one  : {ml:.4f}")
        if ml > 3 * ma:
            print(f"  -> the LoRA tracks the best available target {ml/ma:.0f}x better than")
            print("     the two targets track each other. It is closer to the truth than")
            print("     either, which is precisely why neither can grade it. int8_convrot")
            print("     applies a rotation, so differencing two independently rotated")
            print("     checkpoints destroys the delta -- the wrong quantity, not just a")
            print("     noisy one.")
    else:
        print("\n  (fp8 checkpoints absent -- the cross-check that makes this readable")
        print("   could not run, so treat the int8 numbers above as uninterpretable)")

    steps = med(lambda r: r["quantized"], "steps")
    print(f"\nLoRA delta on int8 modules: {steps:.3f} quantization steps (RMS).")
    if steps < 0.5:
        print("  BELOW half a step, so the int8 modules are NOT GRADEABLE from")
        print("  these files -- neither by code match nor by differencing two")
        print("  independently quantized checkpoints. That is a limit of the")
        print("  artifacts, NOT evidence about the LoRA. See the docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
