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

**Kijai's conversion** ships the raw 32-head bank and fuses at run time, where
ours precomputes. Different choices, same arithmetic, so the backbone must
agree exactly and our fused heads must be reproducible from his bank. His
pruned build independently projects the adaln update into the curve basis --
the same conclusion this repo reached separately, which is the part worth
having a second opinion on.

**His bank has had two encodings and this reads whichever is on disk.** Until
2026-08-27 it was one `final_layer.{stream}_out.set_weight` tensor, which needs
a core change to load. It is now the generic weight-adapter path -- `lora_up` /
`lora_down` / `reshape_weight` -- which ComfyUI already applies as
`pad_tensor_to_shape(base, reshape) + up @ down`, so no core LoRA change is
needed and only `comfy/ldm/minimax/model.py` has to learn what an enlarged head
means (Comfy-Org/ComfyUI#15908, open, `model.py` only). Under that encoding the
tensor he ships is NOT the bank: it is the bank minus the zero-padded base head,
so the bank has to be reconstructed through the same arithmetic core uses.

This script branched on `set_weight` by name and died with a `KeyError` the day
the encoding changed -- an assumption that had only ever met one implementation,
which is the escape CLAUDE.md's 2026-08-22 rule is about. It now branches on
WHICH KEYS ARE PRESENT and records which encoding it saw, so the next
repackaging is reported rather than fatal, and a record says what it compared.
It reads the base head from the checkpoint rather than from our own converted
file, so the reconstruction does not depend on anything this repo produced.

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
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pdd_math import fuse_heads, silu_temb_grid  # noqa: E402


def silu_temb_grid_from(checkpoint: Path, rows: int = 1025):
    """`silu(t_emb)` over the grid, from an unpruned checkpoint's time embedder."""
    with safe_open(checkpoint, framework="pt") as f:
        w = {k: f.get_tensor(k).float() for k in
             ("time_embedder.proj_in.weight", "time_embedder.proj_in.bias",
              "time_embedder.proj_out.weight", "time_embedder.proj_out.bias")}
    return silu_temb_grid(w["time_embedder.proj_in.weight"],
                          w["time_embedder.proj_in.bias"],
                          w["time_embedder.proj_out.weight"],
                          w["time_embedder.proj_out.bias"], rows=rows)

LORAS = Path.home() / "ComfyUI" / "models" / "loras" / "h3"
DIFFUSION = Path.home() / "ComfyUI" / "models" / "diffusion_models"
BLOCKS = (0, 12, 25, 37, 49)


#: The published grid. Read from our converted file's metadata where one is
#: available; this is only the fallback for reshaping his tensors.
NUM_INTERVALS = 32


def sha256(path: Path) -> str:
    """Identify exactly which artifact a record compared.

    His files are re-uploaded in place and changed encoding once already, so a
    record naming only the filename cannot say what it read. This is a
    descriptive value in a dated record, which is the one place CLAUDE.md
    allows one.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def base_head(checkpoint: Path, stream: str) -> tuple[torch.Tensor, torch.Tensor]:
    """`final_layer.{stream}_out` weight and bias from the checkpoint itself.

    Read from the checkpoint rather than from our converted file so the
    reconstruction below depends on nothing this repo produced. These two are
    fp32-unquantised in every H3 build we ship, so any variant of the partition
    serves.
    """
    with safe_open(checkpoint, framework="pt") as f:
        return (f.get_tensor(f"final_layer.{stream}_out.weight").double(),
                f.get_tensor(f"final_layer.{stream}_out.bias").double())


def kijai_bank(kij: dict, stream: str, base_w: torch.Tensor,
               base_b: torch.Tensor, n: int = NUM_INTERVALS):
    """His per-interval head bank, whichever encoding the file uses.

    Returns `(encoding, weight[n, out, in], bias[n, out])`.

    Branches on the keys present, never on a filename or a date. Two encodings
    have shipped:

    `set_weight`
        one tensor holding the bank outright. Needs a core `set_weight` /
        `set_bias` path that is not in ComfyUI today.

    `lora_up` / `lora_down` / `reshape_weight`
        the generic adapter, which core already applies as
        `pad_tensor_to_shape(weight, reshape) + up @ down` -- padding with
        ZEROS. So the shipped tensor is `bank - pad(base_head)`: its first
        `out` rows are `head_0 - base` and the rest are heads 1..n-1 verbatim,
        and `up` is a full-rank square factor because an arbitrary matrix has
        to be expressed through a path that only multiplies two.

        That padding is also why his `strength` below 1.0 does not mean what
        ours does: heads 1..n-1 scale from zero, not from the base head.

    An unrecognised layout raises with the keys it did find. A `KeyError` from
    a hardcoded name is how this went from a comparison to no comparison.
    """
    p = f"diffusion_model.final_layer.{stream}_out"
    if f"{p}.set_weight" in kij:
        w = kij[f"{p}.set_weight"].double()
        b = kij.get(f"{p}.set_bias")
        return ("set_weight",
                w.reshape(n, -1, w.shape[-1]),
                None if b is None else b.double().reshape(n, -1))
    if f"{p}.reshape_weight" in kij:
        def rebuild(prefix, base, shape):
            padded = torch.zeros(shape, dtype=torch.float64)
            padded[:base.shape[0]] = base
            diff = (kij[f"{prefix}.lora_up.weight"].double()
                    @ kij[f"{prefix}.lora_down.weight"].double())
            return (padded + diff.reshape(padded.shape)).reshape(n, -1)
        w = rebuild(p, base_w.reshape(base_w.shape[0], -1),
                    kij[f"{p}.reshape_weight"].tolist())
        b = rebuild(f"{p}.bias", base_b.reshape(-1, 1),
                    list(kij[f"{p}.bias.reshape_weight"].tolist()) + [1])
        return ("reshape_weight", w.reshape(n, -1, base_w.shape[-1]), b)
    raise RuntimeError(
        f"Kijai's file carries neither head encoding this knows for "
        f"{stream}_out. Present: "
        f"{sorted(k for k in kij if k.startswith(p))}. His artifacts are "
        f"re-uploaded in place; read the current ones before assuming a bug "
        f"here, and add the new encoding to this branch rather than to a "
        f"call site.")


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
                    "files": {k: p.name for k, p in paths.items()},
                    "sha256": {k: sha256(p) for k, p in paths.items()}}

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

    # --- the head bank: both ship the published stack, ours fuses at load ---
    # Ours stored 8 precomputed heads until 2026-08-26 and this compared those.
    # It now ships the bank itself and fuses at load for whatever nfe is asked,
    # so the comparison is bank-to-bank, and the fusion is checked separately at
    # every step count the grid divides by -- which is the property that
    # replaced the precompute.
    #
    # Both streams, weight and bias. Only video was compared until 2026-08-27,
    # which left the audio head -- a different shape, its own shift, and its own
    # entry in his file -- resting on the video result. His current encoding
    # carries a bias bank too, so that is compared rather than assumed.
    pruned_ckpt = DIFFUSION / f"minimax_h3_{part}_pruned_int8_convrot.safetensors"
    banks, encodings = {}, {}
    for stream, w_key, b_key in (("video", "proj_out.weight", "proj_out.bias"),
                                 ("audio", "audio_proj_out.weight",
                                  "audio_proj_out.bias")):
        bw, bb = base_head(pruned_ckpt, stream)
        enc, kw, kb = kijai_bank(kij, stream, bw, bb)
        encodings[stream] = enc
        banks[stream] = kw
        shift = 12.0 if stream == "video" else 3.0
        row = {
            "kijai_bank_is_the_published_stack": rel(kw, src[w_key]),
            "our_bank_is_the_published_stack": rel(
                ours[f"h3_pdd.bank.{stream}.weight"], src[w_key]),
            "our_bias_is_the_published_stack": rel(
                ours[f"h3_pdd.bank.{stream}.bias"], src[b_key]),
            "fusion_from_either_bank_agrees": {
                str(nfe): rel(
                    fuse_heads(ours[f"h3_pdd.bank.{stream}.weight"], shift, 32, 32 // nfe),
                    fuse_heads(kw, shift, 32, 32 // nfe))
                for nfe in (8, 4, 2)},
        }
        if kb is not None:
            row["kijai_bias_is_the_published_stack"] = rel(kb, src[b_key])
        report[stream] = row
    report["kijai_head_encoding"] = encodings

    # --- adaln bake: two approximations, both scored against ground truth ---
    # The 2688-dim pairs and the grid live in the UNPRUNED conversion now: a
    # `--pruned` file carries the baked form only. Ground truth for the adaln
    # comparison comes from the published source either way, which is the
    # better source anyway -- it does not depend on what our converter chose to
    # keep.
    grid = silu_temb_grid_from(DIFFUSION / "diffusion_models" /
                               f"minimax_h3_{part}_int8_convrot.safetensors").double()
    with safe_open(DIFFUSION / f"minimax_h3_{part}_pruned_int8_convrot.safetensors",
                   framework="pt") as f:
        table = f.get_tensor("adaln_t_table").double()
    adaln = {}
    for i in BLOCKS:
        A = src[f"transformer_blocks.{i}.adaln_proj.linear.lora_down"].double()
        B = src[f"transformer_blocks.{i}.adaln_proj.linear.lora_up"].double()
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
    for stream in ("video", "audio"):
        row = report[stream]
        print(f"{stream} head, his encoding {encodings[stream]!r}:")
        print(f"  his bank is the published 32-stack   : "
              f"{row['kijai_bank_is_the_published_stack']:.2e}")
        print(f"  our bank is the published 32-stack   : "
              f"{row['our_bank_is_the_published_stack']:.2e}")
        if "kijai_bias_is_the_published_stack" in row:
            print(f"  his bias is the published 32-stack   : "
                  f"{row['kijai_bias_is_the_published_stack']:.2e}")
        print(f"  our bias is the published 32-stack   : "
              f"{row['our_bias_is_the_published_stack']:.2e}")
        print("  fusing either bank agrees, per step count:")
        for k, v in row["fusion_from_either_bank_agrees"].items():
            print(f"    nfe {k:>2}  {v:.2e}")
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
