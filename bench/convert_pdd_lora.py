#!/usr/bin/env python3
"""Convert an alibaba-pai MiniMax-H3 PDD acceleration LoRA to this repo's format.

Parallel Decoding Distillation (PDD) is not a step-distillation LoRA and does
not load through `LoraLoaderModelOnly`. One published file holds three
mechanisms that reach the model on three different surfaces:

  1. 312 backbone LoRA modules (attn + MLP, 50 blocks + 2 refiner blocks).
     A weight patch, once the keys are ComfyUI's.
  2. 50 `adaln_proj.linear` LoRA modules. A weight patch on an UNPRUNED
     checkpoint; on a pruned one the 2688-dim time-embedding space it lives in
     does not exist, and it has to be re-injected at run time.
  3. `proj_out` and `audio_proj_out` replaced by 32 stacked per-interval heads.
     Not a delta at all, and not expressible as any kind of LoRA.

This script does every transform that can be done once, offline, and leaves the
node with only what genuinely needs the running model. Its output is one file
that `MiniMaxH3PDDLoRA` reads.

## What it emits, and why each piece is here

`diffusion_model.*.lora_A/lora_B/alpha`
    The backbone, in ComfyUI generic-LoRA naming, so `comfy.lora.load_lora`
    applies it through the supported path rather than one we maintain. Four
    transforms, each verified numerically against the release weights before
    this script was written (see `bench/results/`):
      - q/k/v fuse into `attn.qkv_proj`: A concatenated, B block-diagonal.
        Rank becomes 3x, so alpha becomes 3x to hold `alpha/rank` fixed.
      - `attn.to_out.0` -> `attn.out_proj`, `ff.net.2` -> `mlp.fc2`: renames.
      - `ff.net.0.proj` -> `mlp.fc1` with the output halves SWAPPED. The
        release stores SwiGLU as [value; gate] and ComfyUI as [gate; value].
        Unswapped the delta lands on the wrong half: rel-Frob 1.41 against
        0.009 swapped, measured on block 0.
      - explicit `.alpha` tensors. ComfyUI reads alpha from a tensor and never
        from `__metadata__` (`comfy/weight_adapter/lora.py`), falling back to a
        scale of 1.0. PDD's alpha/rank IS 1.0, so the fallback happens to be
        right -- which is exactly the kind of coincidence
        `bench/check_lora_alpha.py` exists to stop us depending on.

`h3_pdd.adaln.blocks.{i}.lora_A/lora_B`
    Deliberately NOT in ComfyUI naming, because whether these become a weight
    patch or a runtime injection is a property of the loaded checkpoint, not of
    this file. The node branches on `use_adaln_curves` -- an observable of the
    model -- and builds whichever form it needs. Naming them
    `diffusion_model.*` here would let `load_lora` silently apply them to an
    unpruned model behind the node's back, and silently drop them on a pruned
    one with nothing but a log line.

`h3_pdd.head.{video,audio}.{weight,bias}`  [nfe, out, in] / [nfe, out]
    The 32 per-interval heads collapsed to the `nfe` fused heads a run actually
    uses. `MiniMaxH3ParallelHead.forward` fuses WEIGHTS, not outputs, and the
    fusion plan depends only on (shift, num_steps, block_size, step) -- all
    fixed at conversion time. So this is the same arithmetic, not an
    approximation, and it turns a 32-head module swap into an 8-entry lookup.
    Computed in float64 and stored float32: the reference casts its plan to the
    weight dtype (bf16) before the einsum, which costs ~1.7e-3 relative on the
    fused head -- the same order as a tenth of the signal the heads carry. Our
    output heads are ComfyUI's fp32 island, so we keep the precision. This
    means our result is not bit-identical to theirs; it is closer to the
    intended arithmetic, not further.

`h3_pdd.bank.{video,audio}.{weight,bias}`   [32, out, in] / [32, out]
    The published per-interval head stack, verbatim and at its published bf16.
    The node fuses it at LOAD time for whatever `nfe` it is asked for, so one
    file serves every block size the grid divides by -- 8 NFE at block 4, 4 NFE
    at block 8, both of which the vendor's README reports rendering at -- with
    no reconversion.

    Fusing at load rather than per forward is the paper's own recommendation
    (section 3.1: "during inference we can avoid the extra compute of an
    enlarged final layer and we only need to hold one fused linear layer per
    block in memory"). The vendor's adapter and Kijai's conversion both fuse
    inside the forward, which is the enlarged final layer the paper says to
    avoid; storing 8 precomputed heads instead, as this converter first did,
    hits the paper's form but pins the step count into the file.

`h3_pdd.adaln_baked.blocks.{i}.diff` / `.diff_b`   [96768, 8] / [96768]
    The adaln update pre-solved into the pruned checkpoint's OWN rank-8 time
    basis, so it becomes an ordinary weight patch instead of a runtime
    injection. Emitted only with `--pruned`, because the basis is that
    checkpoint's.

    This is possible because the delta's time curve turns out to lie in that
    basis. Measured 2026-08-26 over the 1025-row grid, reconstructed against
    the true delta: **1.2e-5 to 6.1e-5 relative**, roughly fifty times below
    bf16's own resolution. The projection is affine -- the basis plus a
    constant column -- because the pruned form is an SVD of the CENTRED time
    curve and the mean lives in the bias.

    The first version of that measurement omitted the centring and reported
    0.93-0.99, i.e. "cannot be baked". What caught it was projecting the BASE
    adaln curve too: it scored just as badly, which cannot be true of a basis
    fitted to it. A positive control is the only reason this exists.

`h3_pdd.adaln_table`  [1025, 8]
    The table the bake was solved against, so the node can confirm the loaded
    checkpoint carries the same one before using the baked path. Verified
    identical between a partition's `int8_convrot` and `fp8_scaled` builds and
    different between partitions, so one bake serves a partition's variants.

`h3_pdd.silu_temb_grid`  [1025, 2688]
    silu(time_embedder(t)) over `linspace(0, 1, 1025)`, derived from `--base`.
    Consumed ONLY on a pruned checkpoint, where it is what mechanism 2 above
    needs. **It is partition-specific.** The fl2va and ref2va time curves
    differ by 7.8% relative, so a grid built from the wrong partition feeds the
    adaln injection a 7.8%-wrong input and nothing errors. Deriving it here,
    from the same checkpoint that supplies the fingerprint below, is what makes
    that impossible rather than merely documented.

`h3_pdd.base_video_out`  [96, 5376]
    `final_layer.video_out.weight` from `--base`, verbatim. The partition
    check. fl2va and ref2va ship IDENTICAL key sets -- the whole silent-success
    failure `docs/h3_ref2v_distillation.md` records -- so a Ref2VA LoRA loads
    onto an fl2va checkpoint with zero unmatched keys and renders. This tensor
    distinguishes them: 5.0% apart, and unquantised in every checkpoint variant
    we ship.

    **Stored as a tensor and compared by distance, not hashed.** The first
    version put a sha256 of it in the metadata, and it fired on the first real
    render against the RIGHT checkpoint: ComfyUI casts on load, and a cast
    changes every bit while moving the value a fraction of a percent. An exact
    hash of a tensor the loader is allowed to transform cannot distinguish
    "wrong partition" from "loaded normally", which makes it a control that
    reports red on correct state -- the thing CLAUDE.md says is worse than no
    control. A relative-Frobenius comparison separates a cast (~0.3%) from a
    partition swap (5%) with an order of magnitude to spare, and it can say how
    far off it was.

## Borrowed patterns

The runtime adaln injection this file prepares for is a reimplementation of the
approach in `ComfyUI-MiniMax-H3-Turbo` (`_inject_adaln_egrid` /
`_make_adaln_forward`), which solved the same pruned-checkpoint problem for the
v4 turbo pack. We do not import it: its bundled grid is fl2va-only, and the arm
we care about is ref2va. The pattern is theirs and is credited at the point of
use in `pdd_lora.py`; the code and the grid are ours.

## What this does NOT do

It does not read `pdd_config.json`. The vendor's own loader resolves that
sidecar and falls back to hardcoded defaults when it is absent -- and it IS
absent from the published repo, so their script runs on defaults while the
authoritative values sit unread in the file's `__metadata__`. We read the
metadata and fail if it is missing, because a future Acc-4Step with a different
grid would otherwise be converted as 32/4 with no error anywhere.

Exit codes: 0 converted and self-checked, 1 a structural assertion failed.

    python bench/convert_pdd_lora.py \
        --pdd  <MiniMax-H3-Ref2VA-Acc-8Step.safetensors> \
        --base <minimax_h3_ref2va_int8_convrot.safetensors> \
        --out  <minimax_h3_ref2va_pdd_8step_comfy.safetensors>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))              # this repo
from pdd_math import fuse_heads, silu_temb_grid   # noqa: E402

CONVERTER_VERSION = "1"

#: The release scheduler configs (`scheduler/`, `audio_scheduler/`) of
#: MiniMaxAI/MiniMax-H3. `apply_pdd_lora` takes its shifts from the live
#: pipeline rather than from the checkpoint, so these are what the published
#: LoRAs were distilled under. They are also this repo's `SIGMA_SHIFT`, which
#: is why a PDD arm moves no shift node -- only the step count.
DEFAULT_SHIFT_VIDEO = 12.0
DEFAULT_SHIFT_AUDIO = 3.0

#: Rows in the derived time grid. Matches `adaln_t_table`'s first dimension in
#: the pruned checkpoints, so a row index means the same t on both paths.
GRID_ROWS = 1025

#: Release name -> ComfyUI name, for the modules that are a plain rename.
_RENAME = {
    "attn.to_out.0": "attn.out_proj",
    "ff.net.2": "mlp.fc2",
}


def read_metadata(path: Path) -> dict:
    """The PDD config, from the file itself. No sidecar, no defaults."""
    with path.open("rb") as handle:
        import struct
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    meta = header.get("__metadata__")
    if not meta:
        raise SystemExit(
            f"{path.name} carries no __metadata__. The PDD grid, block size, "
            f"rank and alpha are only recorded there; refusing to guess them.")
    required = ("pdd_num_steps", "pdd_block_size", "lora_rank", "lora_alpha",
                "lora_targets")
    missing = [k for k in required if k not in meta]
    if missing:
        raise SystemExit(f"{path.name} __metadata__ is missing {missing}.")
    return meta


def derive_silu_temb_grid(base: Path, rows: int = GRID_ROWS) -> torch.Tensor:
    """`silu(time_embedder(t))` for `t = linspace(0, 1, rows)`, from `base`.

    Reads the four time-embedder tensors off a safetensors handle and hands
    them to `pdd_math.silu_temb_grid`, so this needs no model instance, no CUDA
    and no dequantisation -- the time embedder is unquantised in every
    checkpoint variant we ship.
    """
    with safe_open(base, framework="pt") as f:
        keys = set(f.keys())
        need = {"time_embedder.proj_in.weight", "time_embedder.proj_in.bias",
                "time_embedder.proj_out.weight", "time_embedder.proj_out.bias"}
        if not need <= keys:
            raise SystemExit(
                f"{base.name} has no time_embedder. Pass the UNPRUNED "
                f"checkpoint of the matching partition -- a pruned one replaced "
                f"the time embedder with an 8-column curve table, which is the "
                f"very thing the grid exists to undo.")
        w = {k: f.get_tensor(k).to(torch.float32) for k in need}

    return silu_temb_grid(w["time_embedder.proj_in.weight"],
                          w["time_embedder.proj_in.bias"],
                          w["time_embedder.proj_out.weight"],
                          w["time_embedder.proj_out.bias"],
                          rows=rows, apply_silu=True)


def base_video_out(base: Path) -> torch.Tensor:
    """`final_layer.video_out.weight` from the checkpoint, for the partition check."""
    with safe_open(base, framework="pt") as f:
        if "final_layer.video_out.weight" not in set(f.keys()):
            raise SystemExit(f"{base.name} has no final_layer.video_out.weight.")
        return f.get_tensor("final_layer.video_out.weight").to(torch.float32)


def convert_backbone(src: dict, prefix_in: str, prefix_out: str, index: int,
                     rank: int, alpha: float, out: dict, seen: set) -> int:
    """One transformer or refiner block's attn+MLP modules. Returns modules written.

    Every source key it reads is added to `seen`, which the caller asserts
    covers the whole file. Counting emitted modules instead would not catch a
    target silently skipped -- the count would simply be lower, and the number
    it is compared against is one this script computed itself.
    """
    def take(name):
        ka = f"{prefix_in}.{index}.{name}.lora_down"
        kb = f"{prefix_in}.{index}.{name}.lora_up"
        seen.add(ka)
        seen.add(kb)
        return src[ka], src[kb]

    written = 0
    dst = f"diffusion_model.{prefix_out}.{index}"

    # q/k/v -> qkv_proj. A concatenated over rank; B block-diagonal, so that
    # B @ A reproduces cat([Bq@Aq, Bk@Ak, Bv@Av], 0) exactly. Verified against
    # the checkpoint: ComfyUI's qkv_proj is cat([to_q; to_k; to_v], 0).
    aq, bq = take("attn.to_q")
    ak, bk = take("attn.to_k")
    av, bv = take("attn.to_v")
    fused_rank = rank * 3
    a_cat = torch.cat([aq, ak, av], dim=0)
    b_blk = torch.zeros(bq.shape[0] + bk.shape[0] + bv.shape[0], fused_rank,
                        dtype=bq.dtype)
    r0 = c0 = 0
    for b in (bq, bk, bv):
        b_blk[r0:r0 + b.shape[0], c0:c0 + rank] = b
        r0 += b.shape[0]
        c0 += rank
    out[f"{dst}.attn.qkv_proj.lora_A.weight"] = a_cat
    out[f"{dst}.attn.qkv_proj.lora_B.weight"] = b_blk
    # alpha scales with rank so that alpha/rank -- the applied scale -- is held.
    out[f"{dst}.attn.qkv_proj.alpha"] = torch.tensor(alpha * 3.0)
    written += 1

    for src_name, dst_name in _RENAME.items():
        a, b = take(src_name)
        out[f"{dst}.{dst_name}.lora_A.weight"] = a
        out[f"{dst}.{dst_name}.lora_B.weight"] = b
        out[f"{dst}.{dst_name}.alpha"] = torch.tensor(alpha)
        written += 1

    # ff.net.0.proj -> mlp.fc1, output halves swapped. The delta's rows follow
    # the base layout, so swapping B's row halves is the whole transform.
    a, b = take("ff.net.0.proj")
    half = b.shape[0] // 2
    if b.shape[0] % 2:
        raise SystemExit(f"{dst}.mlp.fc1 lora_up has odd rows {b.shape[0]}; "
                         f"a SwiGLU projection cannot.")
    out[f"{dst}.mlp.fc1.lora_A.weight"] = a
    out[f"{dst}.mlp.fc1.lora_B.weight"] = torch.cat([b[half:], b[:half]], dim=0)
    out[f"{dst}.mlp.fc1.alpha"] = torch.tensor(alpha)
    written += 1
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdd", required=True, type=Path,
                    help="the published PDD LoRA (…-Acc-8Step.safetensors)")
    ap.add_argument("--base", required=True, type=Path,
                    help="UNPRUNED ComfyUI checkpoint of the matching "
                         "partition; supplies the time grid and the "
                         "partition fingerprint")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pruned", type=Path, default=None,
                    help="pruned checkpoint of the SAME partition. Supplies "
                         "`adaln_t_table`, which lets the adaln update be "
                         "pre-solved into that basis and applied as an "
                         "ordinary weight patch instead of a runtime "
                         "injection. Without it the file still works, through "
                         "the slower injection path.")

    ap.add_argument("--shift-video", type=float, default=DEFAULT_SHIFT_VIDEO)
    ap.add_argument("--shift-audio", type=float, default=DEFAULT_SHIFT_AUDIO)
    args = ap.parse_args(argv)

    meta = read_metadata(args.pdd)
    num_steps = int(meta["pdd_num_steps"])
    block_size = int(meta["pdd_block_size"])
    rank = int(meta["lora_rank"])
    alpha = float(meta["lora_alpha"])
    targets = [t for t in str(meta["lora_targets"]).split(",") if t]
    if block_size < 1 or num_steps % block_size:
        raise SystemExit(f"pdd_num_steps={num_steps} is not divisible by "
                         f"pdd_block_size={block_size}.")
    nfe = num_steps // block_size

    expected = {"to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2",
                "adaln_proj.linear"}
    if set(targets) != expected:
        raise SystemExit(
            f"lora_targets {sorted(targets)} is not the layout this converter "
            f"was written against ({sorted(expected)}). Every mapping below "
            f"was verified for that layout and none of them transfer blind.")

    src = load_file(args.pdd, device="cpu")
    out: dict[str, torch.Tensor] = {}

    n_blocks = 1 + max(int(k.split(".")[1]) for k in src
                       if k.startswith("transformer_blocks."))
    n_refiner = 1 + max(int(k.split(".")[2]) for k in src
                        if k.startswith("token_refiner.refiner_blocks."))

    seen: set[str] = set()
    modules = 0
    for i in range(n_blocks):
        modules += convert_backbone(src, "transformer_blocks", "blocks", i,
                                    rank, alpha, out, seen)
        # adaln stays in a neutral namespace: which surface it reaches is a
        # property of the loaded checkpoint, not of this file.
        for suffix, dst in (("lora_down", "lora_A"), ("lora_up", "lora_B")):
            k = f"transformer_blocks.{i}.adaln_proj.linear.{suffix}"
            seen.add(k)
            out[f"h3_pdd.adaln.blocks.{i}.{dst}"] = src[k]
        adaln_modules = i + 1
    for i in range(n_refiner):
        modules += convert_backbone(src, "token_refiner.refiner_blocks",
                                    "token_refiner.blocks", i, rank, alpha, out,
                                    seen)

    # Parallel heads -> the fused heads a run at these shifts actually uses.
    head_keys = ("proj_out.weight", "proj_out.bias",
                 "audio_proj_out.weight", "audio_proj_out.bias")
    seen.update(head_keys)

    # Every tensor in the published file must have been consumed by exactly one
    # of the three mechanisms. A target this converter does not know about --
    # a future variant adding one, or a rename upstream -- lands here as a
    # named leftover rather than as a quietly weaker LoRA.
    leftover = sorted(set(src) - seen)
    if leftover:
        raise SystemExit(
            f"{len(leftover)} source tensors were not consumed, e.g. "
            f"{leftover[:4]}. Every key must reach the backbone, the adaln "
            f"sidecar or the fused heads.")

    # The raw bank, so the node can fuse for any block size the grid divides
    # by. Kept at the published bf16: the fusion runs in float64 from these
    # exact values either way, so widening here would store precision the
    # source never had.
    out["h3_pdd.bank.video.weight"] = src["proj_out.weight"]
    out["h3_pdd.bank.video.bias"] = src["proj_out.bias"]
    out["h3_pdd.bank.audio.weight"] = src["audio_proj_out.weight"]
    out["h3_pdd.bank.audio.bias"] = src["audio_proj_out.bias"]

    grid = derive_silu_temb_grid(args.base)
    out["h3_pdd.base_video_out"] = base_video_out(args.base)

    # The output targets the base you named, and carries one adaln form only.
    #
    # Both were shipped together for a few hours and it was 637 MB of pairs a
    # pruned checkpoint cannot use plus 11 MB of grid the bake makes redundant
    # -- around 40% of the file, dead in the only configuration this repo
    # renders. `--pruned` says which base this is for, so it also says which
    # form to emit; a second flag would have been a way to get them out of step.
    baked = 0
    if args.pruned is None:
        # No pruned base named: the update stays in the 2688-dim time space and
        # the grid comes with it, for the runtime injection.
        out["h3_pdd.silu_temb_grid"] = grid
    else:
        with safe_open(args.pruned, framework="pt") as f:
            if "adaln_t_table" not in set(f.keys()):
                raise SystemExit(
                    f"{args.pruned.name} carries no adaln_t_table, so it is not "
                    f"a pruned/curve-form checkpoint and there is no basis to "
                    f"solve into. Pass the pruned build of this partition.")
            table = f.get_tensor("adaln_t_table").to(torch.float64)
        out["h3_pdd.adaln_table"] = table.to(torch.float32)
        # Affine design matrix: the basis, plus a constant column for the mean
        # the pruned form keeps in its bias rather than in the basis.
        design = torch.cat([table, torch.ones(table.shape[0], 1,
                                              dtype=torch.float64)], dim=1)
        worst = 0.0
        for i in range(n_blocks):
            a = out[f"h3_pdd.adaln.blocks.{i}.lora_A"].to(torch.float64)
            b = out[f"h3_pdd.adaln.blocks.{i}.lora_B"].to(torch.float64)
            curve = grid.to(torch.float64) @ a.T          # [rows, rank]
            coef = torch.linalg.lstsq(design, curve).solution   # [9, rank]
            # fp16: `blocks.N.adaln_proj.linear.weight` is F16 in the pruned
            # checkpoints, so this is the dtype it will be added to. Storing
            # fp32 was half this file's growth for precision the target
            # discards on contact.
            out[f"h3_pdd.adaln_baked.blocks.{i}.diff"] = \
                (b @ coef[:-1].T).to(torch.float16)
            out[f"h3_pdd.adaln_baked.blocks.{i}.diff_b"] = \
                (b @ coef[-1]).to(torch.float32)
            # Graded per block against the delta it replaces, not assumed.
            #
            # This asks "does this curve fit a rank-8 smooth-time basis at
            # all", and it is NOT a partition check -- measured 2026-08-26 by
            # deliberately baking ref2va's delta against fl2va's table, which
            # fit just as well and was written without complaint. Both bases
            # are SVDs of very similar smooth time curves, so they span nearly
            # the same subspace; what differs is the COORDINATES, and a fit is
            # blind to that by construction. The wrong-basis bake is 0.0205
            # wrong at runtime against 0.0001 for the right one.
            #
            # The guard for that is `MiniMaxH3PDDLoRA`'s comparison of
            # `h3_pdd.adaln_table` against the loaded checkpoint's own, which
            # is why this file stores the table rather than trusting the fit.
            true = curve @ b.T
            got = (design @ coef) @ b.T
            err = float((got - true).norm() / true.norm())
            worst = max(worst, err)
            if err > 1e-3:
                raise SystemExit(
                    f"block {i}: the adaln delta does not fit the pruned "
                    f"checkpoint's rank-8 time basis (rel {err:.2e}). Baking it "
                    f"would misapply the modulation at every timestep. Convert "
                    f"without --pruned to keep the runtime injection.")
            baked += 1
        for i in range(n_blocks):
            # The 2688-dim pairs are what the bake replaces. Keeping them would
            # be two representations of one update in one file, and nothing
            # would say which the node used.
            out.pop(f"h3_pdd.adaln.blocks.{i}.lora_A", None)
            out.pop(f"h3_pdd.adaln.blocks.{i}.lora_B", None)
        print(f"  baked {baked} adaln modules into {args.pruned.name}'s basis, "
              f"worst reconstruction {worst:.2e}; dropped the 2688-dim pairs "
              f"and the grid this base cannot use")

    # --- self-check: the emitted delta must equal the source delta ----------
    # Not a restatement of the code above: it reconstructs B @ A on both sides
    # and compares, so a wrong slice in the block-diagonal or a swap applied to
    # the wrong axis fails here rather than in a render.
    i = n_blocks - 1
    ref = torch.cat([src[f"transformer_blocks.{i}.attn.to_{x}.lora_up"].float()
                     @ src[f"transformer_blocks.{i}.attn.to_{x}.lora_down"].float()
                     for x in "qkv"], dim=0)
    got = (out[f"diffusion_model.blocks.{i}.attn.qkv_proj.lora_B.weight"].float()
           @ out[f"diffusion_model.blocks.{i}.attn.qkv_proj.lora_A.weight"].float())
    err = ((got - ref).norm() / ref.norm()).item()
    if err > 1e-6:
        raise SystemExit(f"qkv fusion self-check failed: rel {err:.3e}")

    b = src[f"transformer_blocks.{i}.ff.net.0.proj.lora_up"].float()
    a = src[f"transformer_blocks.{i}.ff.net.0.proj.lora_down"].float()
    half = b.shape[0] // 2
    ref = torch.cat([b[half:], b[:half]], dim=0) @ a
    got = (out[f"diffusion_model.blocks.{i}.mlp.fc1.lora_B.weight"].float()
           @ out[f"diffusion_model.blocks.{i}.mlp.fc1.lora_A.weight"].float())
    err = ((got - ref).norm() / ref.norm()).item()
    if err > 1e-6:
        raise SystemExit(f"fc1 swap self-check failed: rel {err:.3e}")


    metadata = {
        "format": "pt",
        "h3_pdd_converter_version": CONVERTER_VERSION,
        "h3_pdd_source": args.pdd.name,
        "h3_pdd_base": args.base.name,
        # Informational only. The node compares the TENSOR by distance; this
        # is here so two converted files can be told apart by eye.
        "base_video_out_sha256": hashlib.sha256(
            out["h3_pdd.base_video_out"].contiguous().numpy().tobytes()).hexdigest(),
        "pdd_num_steps": str(num_steps),
        "pdd_block_size": str(block_size),
        "adaln_baked_blocks": str(baked),
        "h3_pdd_pruned_base": args.pruned.name if args.pruned else "",
        "pdd_nfe": str(nfe),
        "pdd_shift_video": repr(args.shift_video),
        "pdd_shift_audio": repr(args.shift_audio),
        "pdd_grid_rows": str(GRID_ROWS),
        "lora_rank": str(rank),
        "lora_alpha": repr(alpha),
        "backbone_modules": str(modules),
        "adaln_modules": str(adaln_modules),
        "source_format": "alibaba-pai PDD (diffusers naming, stacked heads)",
        "target_format": "ComfyUI generic LoRA + h3_pdd.* sidecar tensors",
        "swi_glu_mapping": "release [value;gate] -> ComfyUI [gate;value]",
        "qkv_fusion": "concat A; block-diagonal B; alpha multiplied by 3",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out), metadata=metadata)

    print(f"wrote {args.out}")
    print(f"  {modules} backbone modules, {adaln_modules} adaln modules "
          f"({len(src)} source tensors, all consumed)")
    print(f"  {nfe} fused head pairs at shift {args.shift_video}/{args.shift_audio}")
    print(f"  time grid {tuple(out['h3_pdd.silu_temb_grid'].shape)} from {args.base.name}")
    print(f"  partition fingerprint {metadata['base_video_out_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
