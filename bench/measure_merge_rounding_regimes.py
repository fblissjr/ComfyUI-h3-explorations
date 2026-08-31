"""Does the rounding regime change the merge penalty, and by how much.

**The question this exists to settle.** Two records published on 2026-08-31
measure the same merge from different rounding regimes, and the day's headline
was quoted from one of them:

    bench/measure_pdd_quant_interaction.py   deterministic RTN
      -> `e_patched` 0.010452 against `e_shipped` 0.009362, the +11.6%
    bench/measure_merge_noise.py             seeded stochastic
      -> `noise/|d|` median 1.981, and 0.921% against the weight

The interaction script says so in its own record --
`"rounding": "deterministic; the shipped path uses seeded stochastic rounding,
so these are its expectation"`. What nobody checked is whether the expectation
is the number, because `bench/measure_stochastic_rounding.py` measured on the
same day that stochastic rounding costs **sqrt(2)** in L2 against RTN for a
fixed tensor. If that sqrt(2) carried into `e_patched`, the shipped merge is
worse than +11.6% and the headline understates it.

**This is not a re-measurement of the same thing with a flag flipped.** The
seed is the one that ships: `comfy/model_patcher.py:928` passes
`seed=comfy.utils.string_to_seed(key)` into the layout's `set_weight`, and that
key is deterministic per module. `bench/measure_merge_noise.py` deliberately
uses one fixed seed across all keys because it is measuring the distribution;
that is a different question and both are wanted.

**What the per-module column is NOT, and this was got wrong here first.** An
earlier draft of this file called the stochastic column "the realisation this
box would produce on a real load". It is not. `comfy_kitchen`'s registry
resolves `quantize_int8_convrot_weight` to the **eager** implementation on CPU
and to a **CUDA** one on GPU, and on the same seed those two draw DIFFERENT
noise -- about a third of the int8 codes differ by one step. A real load runs
the CUDA one. What survives, and what `cross_backend` in the record checks
rather than assumes, is that the two agree on MAGNITUDE to within 0.01% and
both sit at exactly sqrt(2) x the round-to-nearest error. So the means here are
the shipped path's; a single module's value is one draw from it.

**Three arms, one strength, all 200 int8 modules.** Only 200 exist -- every
`I8` weight in the checkpoint is a `blocks.*` linear, `token_refiner` is BF16
and `adaln_proj.linear` is F16 (checked at write time below), so this is the
whole population that requantises rather than a sample of it.

    e_shipped        Q(W_ref)        vs W_ref          what the base carries
    e_merged_rtn     Q(Q(W_ref)+d)   vs W_ref + d      the published +11.6%
    e_merged_stoch   same, shipped per-key seed        what a load produces
    e_baked_rtn      Q(W_ref + d)    vs W_ref + d      one rounding, offline

`e_baked_rtn` is carried because it is the arm the merge is being judged
against and because reproducing it here from a different implementation is the
control on the published 0.009362.

**The control that makes the stochastic column meaningful.** The RTN column is
computed twice -- once through `measure_pdd_quant_interaction.py`'s hand-rolled
rotate/amax/round, once through `comfy_kitchen`'s
`quantize_int8_convrot_weight(stochastic_rounding=None)` -- and they must
agree. Without it, a difference between the RTN and stochastic columns could be
an implementation difference rather than a rounding one, since only the second
implementation can take a seed. `agreement` in the record is that check.

    python bench/measure_merge_rounding_regimes.py \
        --lora <models>/loras/h3/minimax_h3_fl2va_pdd_8step_comfy.safetensors \
        --base <models>/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
        --reference <release>/FL2VA/transformer \
        --out bench/results/2026-08-31_merge_rounding_regimes.json

**What it is not.** A stored-weight distance, like everything else in this
lane. `int8_convrot` is W8A8 and the activation rounding is untouched here;
`docs/open_experiments.md` #23 is that half. Nothing here was rendered.
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
    Reference, hf_to_comfy, head_dim, marker, stats, weight_in_compute_space)

sys.path.insert(0, str(_HERE.parents[2]))
import comfy.utils  # noqa: E402
from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    _build_hadamard, _rotate_weight, dequantize_int8_convrot_weight,
    quantize_int8_convrot_weight)

KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def requant_handrolled(w: np.ndarray, gs: int) -> np.ndarray:
    """`measure_pdd_quant_interaction.py`'s RTN path, verbatim.

    Kept as a second implementation rather than imported so the agreement
    check below compares two bodies of code and not one called twice.
    """
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    h = _build_hadamard(gs, dtype=torch.float32)
    rot = _rotate_weight(t, h, gs)
    row = (rot.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-30)
    q = torch.clamp(torch.round(rot / row), -127, 127).to(torch.int8)
    return dequantize_int8_convrot_weight(q, row, gs).numpy()


def requant_kitchen(w: np.ndarray, gs: int, seed: int | None) -> np.ndarray:
    """The shipped quantiser, with the seed the shipped path would pass.

    `seed=None` is round-to-nearest (`_round_int8` takes the stochastic branch
    only when the seed is not None AND positive).
    """
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    q, s = quantize_int8_convrot_weight(t, gs, stochastic_rounding=seed)
    return dequantize_int8_convrot_weight(q, s, gs).numpy()


def lora_delta(f, prefix: str) -> np.ndarray:
    a = f.get_tensor(prefix + ".lora_A.weight").to(torch.float32)
    b = f.get_tensor(prefix + ".lora_B.weight").to(torch.float32)
    alpha = float(f.get_tensor(prefix + ".alpha").item())
    return ((alpha / a.shape[0]) * (b @ a)).numpy()


def cross_backend_control(gs: int = 256, seed: int = 4242424) -> dict:
    """Do the eager and CUDA stochastic quantisers agree, and on what.

    Load-bearing, not a formality. Everything measured here runs the eager
    implementation because it runs on CPU; a real ComfyUI load runs whatever
    `comfy_kitchen`'s registry picks for the weight's device, and the two are
    separate bodies of code. CLAUDE.md's rule is that an assumption which has
    only ever met one implementation is not a tested assumption, so this meets
    the other one and records exactly which claim survives contact with it.

    Returns the codes comparison (expected to DIFFER -- different draw), the
    magnitude comparison (expected to agree), and each arm's ratio to
    round-to-nearest (expected sqrt(2)).
    """
    if not torch.cuda.is_available():
        return {"ran": False, "why": "no CUDA device visible to this process"}
    from comfy_kitchen.backends.cuda import (  # noqa: PLC0415
        quantize_int8_convrot_weight as cuda_quant)
    torch.manual_seed(0)
    w = torch.randn(2048, 5376, dtype=torch.float32)
    qe, se = quantize_int8_convrot_weight(w.clone(), gs, stochastic_rounding=seed)
    qc, sc = cuda_quant(w.clone().cuda(), gs, stochastic_rounding=seed)
    qc, sc = qc.cpu(), sc.cpu()
    qr, sr = quantize_int8_convrot_weight(w.clone(), gs, stochastic_rounding=None)

    def rel(q, s):
        d = dequantize_int8_convrot_weight(q, s, gs).float()
        return float((d - w).double().norm() / w.double().norm())

    e_eager, e_cuda, e_rtn = rel(qe, se), rel(qc, sc), rel(qr, sr)
    ndiff = int((qe.to(torch.int16) != qc.to(torch.int16)).sum())
    return {
        "ran": True,
        "shape": list(w.shape), "group_size": gs, "seed": seed,
        "codes_identical": ndiff == 0,
        "codes_differing": ndiff, "codes_total": int(qe.numel()),
        "rel_l2_eager": e_eager, "rel_l2_cuda": e_cuda, "rel_l2_rtn": e_rtn,
        "magnitude_agreement": abs(e_eager - e_cuda) / e_eager,
        "eager_over_rtn": e_eager / e_rtn,
        "cuda_over_rtn": e_cuda / e_rtn,
        "reading": ("same seed, different draw, same magnitude. The per-module "
                    "stochastic column below is one draw from the shipped "
                    "distribution rather than the value a GPU load would "
                    "produce; the means are the shipped path's."),
    }


def int8_population(path: str, hdr: dict) -> dict:
    """Which weights in this checkpoint are actually int8.

    Written into the record because the claim "200 modules is the whole
    population" is otherwise a sentence, and this lane has already had one
    module count read two ways.
    """
    fams: dict[str, int] = {}
    for k, v in hdr.items():
        if k == "__metadata__" or not k.endswith(".weight"):
            continue
        if v.get("dtype") != "I8":
            continue
        fam = "blocks.*" if k.startswith("blocks.") else k.split(".")[0] + ".*"
        fams[fam] = fams.get(fam, 0) + 1
    return fams


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--strength", default=1.0, type=float)
    args = ap.parse_args()

    hdr, off = header(args.base)
    ref = Reference(args.reference)
    hd = head_dim(hdr)
    pop = int8_population(args.base, hdr)
    # Run the control FIRST: if the two backends disagree on magnitude, the
    # 200-module sweep below is measuring a rounding this box does not run.
    cross = cross_backend_control()
    print(f"  cross-backend control: {cross}\n", flush=True)

    n_blocks = 1 + max(int(k.split(".")[1]) for k in hdr
                       if k.startswith("blocks.") and k.endswith(".comfy_quant"))
    blocks = (list(range(n_blocks)) if args.blocks == "all"
              else [int(x) for x in args.blocks.split(",")])
    s = args.strength

    rows = []
    worst_agree = 0.0
    with safe_open(args.lora, "pt") as f:
        for blk in blocks:
            for kind in KINDS:
                mod = f"blocks.{blk}.{kind}"
                gs = int((marker(args.base, hdr, off, mod) or {})
                         ["convrot_groupsize"])
                w_q = weight_in_compute_space(args.base, hdr, off, mod)
                w_ref = hf_to_comfy(mod + ".weight",
                                   ref.get(mod + ".weight"), hd)
                d = lora_delta(f, f"diffusion_model.{mod}") * s
                target = w_ref + d
                merged_in = w_q + d

                # The seed the shipped path would pass for this module.
                seed_key = f"diffusion_model.{mod}.weight"
                seed = int(comfy.utils.string_to_seed(seed_key))

                m_rtn_a = requant_handrolled(merged_in, gs)
                m_rtn_b = requant_kitchen(merged_in, gs, None)
                # Two implementations of the same rounding: if these disagree,
                # the RTN/stochastic comparison below is not attributable.
                agree = float(np.abs(m_rtn_a - m_rtn_b).max())
                worst_agree = max(worst_agree, agree)

                row = {
                    "block": blk, "kind": kind, "groupsize": gs, "seed": seed,
                    "pdd_rel": float(np.linalg.norm(d.astype(np.float64))
                                     / np.linalg.norm(w_ref.astype(np.float64))),
                    "e_shipped": stats(w_ref, w_q)["rel_delta"],
                    "e_merged_rtn": stats(target, m_rtn_a)["rel_delta"],
                    "e_merged_stoch": stats(
                        target, requant_kitchen(merged_in, gs, seed))["rel_delta"],
                    "e_baked_rtn": stats(
                        target, requant_kitchen(target, gs, None))["rel_delta"],
                    "rtn_impl_max_abs_disagreement": agree,
                }
                rows.append(row)
                print(f"  {mod:28s} shipped {row['e_shipped']:.6f}  "
                      f"rtn {row['e_merged_rtn']:.6f}  "
                      f"stoch {row['e_merged_stoch']:.6f}  "
                      f"baked {row['e_baked_rtn']:.6f}", flush=True)
                del w_q, w_ref, d, target, merged_in, m_rtn_a, m_rtn_b

    def mean(k: str) -> float:
        return float(np.mean([r[k] for r in rows]))

    base_e = mean("e_shipped")
    summary = {
        "modules": len(rows),
        "e_shipped_mean": base_e,
        "e_merged_rtn_mean": mean("e_merged_rtn"),
        "e_merged_stoch_mean": mean("e_merged_stoch"),
        "e_baked_rtn_mean": mean("e_baked_rtn"),
        "inflation_merged_rtn": mean("e_merged_rtn") / base_e,
        "inflation_merged_stoch": mean("e_merged_stoch") / base_e,
        "inflation_baked_rtn": mean("e_baked_rtn") / base_e,
        "stoch_over_rtn_on_the_merge_gap":
            ((mean("e_merged_stoch") - base_e) / (mean("e_merged_rtn") - base_e)),
        "modules_where_stoch_worse":
            int(sum(r["e_merged_stoch"] > r["e_merged_rtn"] for r in rows)),
    }

    out = {
        "measured": _dt.date.today().isoformat(),
        "produced_by": "bench/measure_merge_rounding_regimes.py",
        "question": ("whether the +11.6% merge inflation, computed under "
                     "deterministic rounding, is the number the shipped "
                     "stochastic path produces"),
        "checkpoint": Path(args.base).name,
        "lora": Path(args.lora).name,
        # Tail only: this record is git-tracked and the release lives
        # outside the repo.
        "reference": "/".join(args.reference.parts[-3:]),
        "strength": s,
        "seed_source": ("comfy.utils.string_to_seed('diffusion_model.<mod>"
                        ".weight'), the key comfy/model_patcher.py:928 passes. "
                        "The MEANS are the shipped path's; a single module's "
                        "stochastic value is one draw, because a GPU load runs "
                        "the CUDA quantiser and it draws differently on the "
                        "same seed. See cross_backend."),
        "cross_backend": cross,
        "int8_population": pop,
        "population_note": ("every I8 weight in this checkpoint is a blocks.* "
                            "linear; token_refiner is BF16 and "
                            "adaln_proj.linear is F16, so neither requantises "
                            "and the 200 here are the whole affected "
                            "population rather than a sample"),
        "agreement": {
            "what": ("max abs elementwise disagreement between the hand-rolled "
                     "RTN of measure_pdd_quant_interaction.py and "
                     "comfy_kitchen's quantize_int8_convrot_weight(seed=None), "
                     "over every module"),
            "worst": worst_agree,
            "ok": worst_agree == 0.0,
            "why": ("without this, an RTN/stochastic difference could be an "
                    "implementation difference; only the second "
                    "implementation takes a seed"),
        },
        "is_not": ("an activation, runtime or output measurement. Stored "
                   "weights only; int8_convrot is W8A8 and the activation "
                   "rounding is untouched. docs/open_experiments.md #23"),
        "summary": summary,
        "modules": rows,
        "reproduce": ("python bench/measure_merge_rounding_regimes.py --lora "
                      "<lora> --base <base> --reference <release>/FL2VA/"
                      "transformer --out <out>"),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n{len(rows)} modules -> {args.out}")
    for k, v in summary.items():
        print(f"  {k:34s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
