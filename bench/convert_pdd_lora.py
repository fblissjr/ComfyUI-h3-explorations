#!/usr/bin/env python3
"""Convert an alibaba-pai MiniMax-H3 PDD acceleration LoRA to this repo's format.

Parallel Decoding Distillation (PDD) is not a step-distillation LoRA and does
not load through `LoraLoaderModelOnly`. One published file holds three
mechanisms that reach the model on three different surfaces:

  1. 312 source LoRA modules (attn + MLP, 50 blocks + 2 refiner blocks),
     emitted as 208 after the q/k/v fuse -- `backbone_modules` in the metadata.
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

`h3_pdd.head.*`  RETIRED 2026-08-27, and deliberately not re-added.
    This held the 32 heads collapsed to `nfe` fused ones. That pinned a step
    count into the artifact, which stopped being knowable here the day the node
    began reading it off the sampler's schedule. The bank below replaced it;
    the node fuses each span it is asked for.

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
    the true delta: **1.2e-5 to 1.1e-4 relative** over all fifty blocks --
    an earlier range of "1.2e-5 to 6.1e-5" was the three blocks sampled by
    hand, and `docs/h3_pdd.md`'s "worst 1.1e-4" is the correct one. Still an
    order of magnitude below the pruning error the modulation already carries. The projection is affine -- the basis plus a
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

`h3_pdd.backbone_probe`  [64, 14336] int8, and `h3_pdd.backbone_probe_scale`  [64, 1] fp32
    The leading rows of `blocks.49.mlp.fc2.weight` AND of its per-row
    `weight_scale`, verbatim, from the ONE checkpoint this file is paired
    with: the base (`--pruned` when given, else `--base`) for a full sidecar,
    the exact baked checkpoint (`--baked`) for a stripped one; metadata
    `h3_pdd_backbone_probe_of` says which. The backbone IDENTITY check: the
    node requires the loaded module to EQUAL both slices, and refuses the
    mismatches that all render normally -- a full sidecar on a checkpoint
    with the backbone already baked in (applied twice), a stripped one on
    the unbaked base (applied never) or on any bake but its own (another
    strength, another LoRA: the wrong backbone). Equality, not distance,
    since converter version 3: version 2 compared codes only and accepted any
    distance above a tolerance on a stripped file, which proved "not the
    base" and let every wrong bake through, and let a scale-only change --
    different arithmetic, same codes -- escape (Codex's 2026-09-03 audit).
    `fc2` of block 49 because it carries the largest single update in the
    file; 64 rows because that is 0.9 MB against 77 MB for the module. When
    `--base` and `--pruned` are both given the two are asserted identical on
    both slices, which is the "pruning touches only adaln" premise made
    checkable. A stripped file also carries the base's pair as
    `h3_pdd.backbone_probe_base` / `_base_scale`, so the node can say
    "this is the unbaked base" rather than "not your bake".

`--omit-backbone --baked <checkpoint>`: the STRIPPED sidecar, for that bake.
    Every `diffusion_model.blocks.*` LoRA tensor is dropped after the
    self-checks have run on them; the refiner, the adaln form, the head bank,
    the tables and the probes stay. `--baked` is REQUIRED: a stripped sidecar
    is paired with one checkpoint and cannot be cut without it, and the
    baked file must differ from the base on the probed module or it is not a
    bake. Metadata `h3_pdd_backbone` says `stripped`,
    `h3_pdd_backbone_strength_baked` records the strength the bake used
    (`--baked-strength`), which the node refuses to differ from, and
    `h3_pdd_baked_checkpoint` names the file. `backbone_modules` counts the
    modules the FILE carries (8: the refiner), `backbone_modules_converted`
    the modules converted (208). Chosen over detecting a bake in the node,
    which would fail open the day the marker is absent; this fails closed on
    the probe either way.

The population is FIXED by the release, not read off the input.
    `vendor_config.transformer_depth()` gives the DiT depth and refiner depth
    from the release's own `transformer/config.json`; times the four backbone
    kinds per block that is 200 + 8 modules, and every count here is asserted
    against it BEFORE anything is emitted. Version 2 derived the expectation
    from the source's own maximum index and from what the loops wrote, so a
    coherent 49-block source would have been converted and stripped as a
    complete file (Codex's 2026-09-03 audit, blocker B).

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
sys.path.insert(0, str(HERE.parent))                # this repo, for vendor_config
import vendor_config  # noqa: E402
sys.path.insert(0, str(HERE.parent))              # this repo
from pdd_math import silu_temb_grid   # noqa: E402

#: "3" since 2026-09-03: the probe carries row scales as well as codes, is cut
#: from the PAIRED checkpoint (`h3_pdd_backbone_probe_of`), and the population
#: is asserted against the release's declared depth. "2" (2026-09-02) probed
#: codes only, from the base, and the node refuses its stripped files. "1"
#: files carry no probe and load as before, unprobed.
CONVERTER_VERSION = "3"

#: The release scheduler configs (`scheduler/`, `audio_scheduler/`) of
#: MiniMaxAI/MiniMax-H3. `apply_pdd_lora` takes its shifts from the live
#: pipeline rather than from the checkpoint, so these are what the published
#: LoRAs were distilled under. They are also this repo's `SIGMA_SHIFT`, which
#: is why a PDD arm moves no shift node -- only the step count.
DEFAULT_SHIFT_VIDEO = 12.0
DEFAULT_SHIFT_AUDIO = 3.0

#: Median `||row_i|| / ||row_0||` over i>=1 in a published head bank, below
#: which the stack is read as DELTAS FROM HEAD 0 rather than verbatim heads.
#:
#: Why a norm and not a key name. `compare_pdd_conversions.kijai_bank` already
#: branches between two encodings, and it does it on the KEYS PRESENT, which is
#: right there: his two layouts ship different tensor names. That method cannot
#: work here. A re-upload of the source would keep `proj_out.weight` exactly and
#: change only what is inside it, so the keys are identical under both encodings
#: and only the values separate them. This is the repo's rule about branching on
#: the observable, and the observable here is the values.
#:
#: Why it separates, measured 2026-08-28 on both published partitions:
#:
#:                        rows1..31/row0     what a delta stack would give
#:   FL2VA  video               1.0010                            0.0289
#:   FL2VA  audio               0.9999                            0.0169
#:   Ref2VA video               1.0038                            0.0290
#:   Ref2VA audio               0.9993                            0.0245
#:
#: Consecutive heads are nearly the same map -- they differ by 2-3% of a head's
#: own norm -- so a delta encoding is not a near miss here, it is a factor of
#: ~35. 0.5 sits about 17x above the largest delta case and 2x below the
#: smallest verbatim one. It is not a precision tolerance and must not be tuned
#: like one: any value in (0.05, 0.9) makes the identical decision on every
#: artifact either encoding can produce.
BANK_VERBATIM_RATIO = 0.5


def bank_row_ratio(w) -> float:
    """Median norm of rows 1.. against row 0's, for an `[n, out, in]` stack."""
    n = w.to(torch.float64).flatten(1).norm(dim=1)
    return float(n[1:].median() / n[0])


def assert_bank_verbatim(src: dict, head_keys) -> dict:
    """Refuse a source whose head stack is deltas rather than verbatim heads.

    The hazard this closes. The converter below copies `proj_out.weight` into
    `h3_pdd.bank.*` UNCHANGED, and nothing downstream would notice if that were
    the wrong thing to do: the partition guard compares the base checkpoint's
    head, not the bank, and `compare_pdd_conversions.py` grades our bank against
    the very file it was copied from, so it agrees with itself under either
    encoding. A delta-encoded re-upload therefore converts silently and renders
    silently, and the first sign is bad output nobody can attribute.

    That is not hypothetical: `Comfy-Org/ComfyUI#15908` changed its head formula
    after `bd016b75ff9b` to one correct only if the stored rows are deltas from
    head 0, and the HF repo's `lastModified` sits two minutes after that commit.
    Our copies are still verbatim -- checked, not assumed, every run from here.

    RAISES rather than converting the deltas itself. Reconstructing them needs
    the upstream convention (is row i `head_i - head_0`, or a running
    difference?), and the file does not say. Guessing would replace a loud
    failure with a quiet wrong bank, which is the thing this exists to prevent.
    """
    ratios = {}
    for key in head_keys:
        if not key.endswith(".weight"):
            continue
        ratios[key] = bank_row_ratio(src[key])
    bad = {k: r for k, r in ratios.items() if r < BANK_VERBATIM_RATIO}
    if bad:
        detail = ", ".join(f"{k} {r:.4f}" for k, r in sorted(bad.items()))
        raise SystemExit(
            f"the published head stack does not look like verbatim heads: "
            f"{detail}, against a verbatim ~1.0 and a threshold of "
            f"{BANK_VERBATIM_RATIO}. Rows 1.. being far smaller than row 0 is "
            f"what a DELTA-FROM-HEAD-0 encoding looks like, and this converter "
            f"copies the stack into `h3_pdd.bank.*` verbatim -- so it would "
            f"produce a wrong bank that nothing downstream checks. Upstream is "
            f"known to have changed this encoding once. Establish the "
            f"convention before converting; do not relax this number.")
    return ratios

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


#: Where `h3_pdd.backbone_probe` is taken from. The node reads both off the
#: file's metadata, never from here, so a file and the node cannot disagree.
BACKBONE_PROBE_KEY = "blocks.49.mlp.fc2.weight"
BACKBONE_PROBE_ROWS = 64

#: Backbone module kinds per block in ComfyUI's layout: `attn.qkv_proj` (the
#: q/k/v fuse), the two `_RENAME` targets, and `mlp.fc1` on the swap path.
#: `bench/check_pdd_sidecar_contract.py` asserts this equals the node's
#: `BACKBONE_KINDS`; times the release's depth it is the fixed population.
BACKBONE_KINDS_PER_BLOCK = 4


def expected_population() -> dict:
    """The fixed population from the release's declared depth, never counted."""
    depth, refiner = vendor_config.transformer_depth()
    return {"blocks": depth, "refiner_blocks": refiner,
            "block_modules": depth * BACKBONE_KINDS_PER_BLOCK,
            "refiner_modules": refiner * BACKBONE_KINDS_PER_BLOCK}


def assert_population(n_blocks: int, n_refiner: int, block_modules: int,
                      refiner_modules: int) -> dict:
    """Refuse a source whose shape is not the release's, before emitting.

    Every argument is what was OBSERVED (indices in the source, modules the
    loops wrote); the expectation comes from `expected_population` alone.
    Returns the expectation for the caller to strip against.
    """
    pop = expected_population()
    got = {"blocks": n_blocks, "refiner_blocks": n_refiner,
           "block_modules": block_modules, "refiner_modules": refiner_modules}
    bad = {k: (got[k], pop[k]) for k in pop if got[k] != pop[k]}
    if bad:
        raise SystemExit(
            f"the source is not the release's population: "
            + ", ".join(f"{k} {g} (release declares {w})"
                        for k, (g, w) in bad.items())
            + ". A coherent short population would otherwise convert as a "
              "complete file and render with modules at their base weights.")
    return pop


def backbone_probe(ckpt: Path, key: str = BACKBONE_PROBE_KEY,
                   rows: int = BACKBONE_PROBE_ROWS) -> tuple[torch.Tensor, torch.Tensor]:
    """(int8 codes, fp32 row scales) for the leading `rows` of one backbone
    weight, as stored in an int8_convrot checkpoint."""
    scale_key = key[: -len(".weight")] + ".weight_scale"
    with safe_open(ckpt, framework="pt") as f:
        keys = set(f.keys())
        for k in (key, scale_key):
            if k not in keys:
                raise SystemExit(f"{ckpt.name} has no {k}; not an int8_convrot "
                                 f"H3 DiT, or a layout this converter does not "
                                 f"know.")
        codes = f.get_slice(key)
        scale = f.get_slice(scale_key)
        if codes.get_dtype() != "I8" or scale.get_dtype() != "F32":
            raise SystemExit(
                f"{ckpt.name}: {key} is {codes.get_dtype()} with a "
                f"{scale.get_dtype()} scale; the probe is I8 codes with F32 "
                f"row scales, exactly as int8_convrot stores them.")
        return codes[:rows].contiguous(), scale[:rows].contiguous()


def probe_digest(codes: torch.Tensor, scale: torch.Tensor) -> str:
    """sha256 over the exact code and scale bytes, informational."""
    h = hashlib.sha256()
    h.update(codes.contiguous().numpy().tobytes())
    h.update(scale.contiguous().numpy().tobytes())
    return h.hexdigest()


def strip_backbone(out: dict, block_modules: int, refiner_modules: int) -> int:
    """Drop every transformer-block backbone tensor from `out`, in place.

    `block_modules` and `refiner_modules` are the RELEASE's counts from
    `expected_population`, never what this run observed -- the version that
    took the counts `convert_backbone` returned let a 49-block source strip
    as a complete file. Asserts the shape of what it removes and of what is
    left against them, so a dict short a kind or long a stray key refuses
    rather than shipping a sidecar whose shape is the shape of what survived.
    Returns the number of tensors removed.

    History: the first version derived kinds-per-block from the rename table
    and expected 450 of the 600; `bench/check_pdd_sidecar_contract.py` went
    red on its first run and so did the first real conversion.
    """
    per_module = 3                                   # lora_A, lora_B, alpha
    keys = [k for k in out if k.startswith("diffusion_model.blocks.")]
    expected = block_modules * per_module
    if len(keys) != expected:
        raise SystemExit(
            f"--omit-backbone: {len(keys)} tensors under "
            f"diffusion_model.blocks.* against {expected} expected "
            f"({block_modules} modules x {per_module}); refusing to strip a "
            f"dict whose shape is already wrong.")
    for k in keys:
        del out[k]
    left = [k for k in out if k.startswith("diffusion_model.")]
    expected_left = refiner_modules * per_module
    if len(left) != expected_left:
        raise SystemExit(
            f"--omit-backbone: {len(left)} diffusion_model.* tensors remain "
            f"against the refiner's {expected_left}; something other than "
            f"the backbone and the refiner was under that prefix.")
    return len(keys)


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

    ap.add_argument("--omit-backbone", action="store_true",
                    help="emit the STRIPPED sidecar for the checkpoint named "
                         "by --baked: every diffusion_model.blocks.* LoRA "
                         "tensor dropped after the self-checks, "
                         "refiner/adaln/heads/probes kept")
    ap.add_argument("--baked", type=Path, default=None,
                    help="with --omit-backbone, REQUIRED: the exact baked "
                         "checkpoint this stripped sidecar is paired with; "
                         "its probe slices are stored for the node to "
                         "require equality against")
    ap.add_argument("--baked-strength", type=float, default=None,
                    help="with --omit-backbone: the strength the bake used, "
                         "recorded for the node to refuse a mismatch "
                         "(default 1.0)")
    ap.add_argument("--shift-video", type=float, default=DEFAULT_SHIFT_VIDEO)
    ap.add_argument("--shift-audio", type=float, default=DEFAULT_SHIFT_AUDIO)
    args = ap.parse_args(argv)
    if args.baked_strength is not None and not args.omit_backbone:
        raise SystemExit("--baked-strength means nothing without "
                         "--omit-backbone: a full sidecar's backbone follows "
                         "the node's strength knob.")
    if args.omit_backbone and args.baked is None:
        raise SystemExit("--omit-backbone needs --baked <checkpoint>: a "
                         "stripped sidecar is paired with ONE baked file and "
                         "cannot be cut without it.")
    if args.baked is not None and not args.omit_backbone:
        raise SystemExit("--baked means nothing without --omit-backbone.")
    baked_strength = (1.0 if args.baked_strength is None
                      else float(args.baked_strength))

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
    block_modules = 0
    for i in range(n_blocks):
        block_modules += convert_backbone(src, "transformer_blocks", "blocks",
                                          i, rank, alpha, out, seen)
        # adaln stays in a neutral namespace: which surface it reaches is a
        # property of the loaded checkpoint, not of this file.
        for suffix, dst in (("lora_down", "lora_A"), ("lora_up", "lora_B")):
            k = f"transformer_blocks.{i}.adaln_proj.linear.{suffix}"
            seen.add(k)
            out[f"h3_pdd.adaln.blocks.{i}.{dst}"] = src[k]
        # The alpha travels WITH the pair. On the unpruned path the node hands
        # these to `comfy.lora` as an ordinary weight patch, and ComfyUI reads
        # alpha from a tensor and never from `__metadata__` -- so without one
        # those 50 modules take a scale of 1.0 while the backbone takes
        # alpha/rank. That is right only while alpha/rank IS 1.0, which is the
        # coincidence the explicit backbone alphas above exist to refuse.
        out[f"h3_pdd.adaln.blocks.{i}.alpha"] = torch.tensor(alpha)
        adaln_modules = i + 1
    refiner_modules = 0
    for i in range(n_refiner):
        refiner_modules += convert_backbone(
            src, "token_refiner.refiner_blocks", "token_refiner.blocks", i,
            rank, alpha, out, seen)
    modules = block_modules + refiner_modules
    # The release's population, not this file's: asserted before anything
    # downstream can treat the observed shape as the expected one.
    pop = assert_population(n_blocks, n_refiner, block_modules, refiner_modules)

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

    # Verbatim heads, or deltas from head 0? Checked BEFORE the copy below,
    # because that copy is what makes the question matter: it moves the stack
    # across unchanged, so a wrong encoding survives into the bank intact.
    bank_ratios = assert_bank_verbatim(src, head_keys)

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
            # The alpha travelled with the pair and is meaningless without it;
            # it was left behind from the day the alpha was added until
            # 2026-09-02, 50 inert tensors the node never read.
            out.pop(f"h3_pdd.adaln.blocks.{i}.alpha", None)
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

    # The backbone identity probe: codes and row scales from the ONE
    # checkpoint this file is paired with.
    base_src = args.pruned if args.pruned is not None else args.base
    base_codes, base_scale = backbone_probe(base_src)
    if args.pruned is not None:
        other_codes, other_scale = backbone_probe(args.base)
        if not (torch.equal(other_codes, base_codes)
                and torch.equal(other_scale, base_scale)):
            raise SystemExit(
                f"{args.base.name} and {args.pruned.name} differ on "
                f"{BACKBONE_PROBE_KEY}[:{BACKBONE_PROBE_ROWS}] or its scale. "
                f"Pruning is documented as touching only adaln_proj, and a "
                f"probe taken from one would misjudge the other; that premise "
                f"is false for these two files and needs looking at before "
                f"anything is emitted.")
    if args.omit_backbone:
        probe_src = args.baked
        probe_codes, probe_scale = backbone_probe(args.baked)
        # The bake stamps the strength it was built at; the flag must agree,
        # or a stripped sidecar records a strength the node then trusts
        # (interrupted review, 2026-09-05). A bake without the stamp (none
        # exists) keeps the flag as the only source.
        with safe_open(args.baked, framework="pt") as _f:
            _stamped = (_f.metadata() or {}).get("h3_bake_strength")
        if _stamped is not None and float(_stamped) != baked_strength:
            raise SystemExit(
                f"{args.baked.name} was baked at strength {_stamped} "
                f"(its h3_bake_strength metadata); --baked-strength is "
                f"{baked_strength:g}. Refusing to cut a sidecar that would "
                f"declare the wrong strength.")
        if torch.equal(probe_codes, base_codes) and torch.equal(probe_scale, base_scale):
            raise SystemExit(
                f"{args.baked.name} equals {base_src.name} on "
                f"{BACKBONE_PROBE_KEY}[:{BACKBONE_PROBE_ROWS}] and its scale, "
                f"so it is not a bake of this backbone. A stripped sidecar cut "
                f"against it would be indistinguishable from one for the base.")
        # The base's pair too, so the node can say "this IS the unbaked base"
        # instead of only "not your bake".
        out["h3_pdd.backbone_probe_base"] = base_codes
        out["h3_pdd.backbone_probe_base_scale"] = base_scale
    else:
        probe_src = base_src
        probe_codes, probe_scale = base_codes, base_scale
    out["h3_pdd.backbone_probe"] = probe_codes
    out["h3_pdd.backbone_probe_scale"] = probe_scale

    # Stripped AFTER the self-checks above, which read the backbone tensors,
    # and against the RELEASE's counts, not this run's.
    stripped = 0
    if args.omit_backbone:
        stripped = strip_backbone(out, pop["block_modules"],
                                  pop["refiner_modules"])
    modules_in_file = modules - (stripped // 3)

    # Provenance the file carries about its own making, so an inventory can
    # date it without trusting a filesystem mtime that a copy resets.
    import datetime as _dt
    import subprocess
    try:
        commit = subprocess.run(
            ["git", "-C", str(HERE.parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(HERE.parent), "status", "--porcelain",
             "--", "bench/convert_pdd_lora.py"],
            capture_output=True, text=True, check=True).stdout.strip()
        if dirty:
            commit += "+dirty"
    except Exception:  # not a checkout, or no git: say so rather than guess
        commit = "unknown"

    metadata = {
        "format": "pt",
        "h3_pdd_converter_version": CONVERTER_VERSION,
        "h3_pdd_converted_on": _dt.date.today().isoformat(),
        "h3_pdd_converter_commit": commit,
        "h3_pdd_source": args.pdd.name,
        "h3_pdd_base": args.base.name,
        # `h3_pdd_base` is the `--base` argument -- the checkpoint the partition
        # check was taken against. For a `--pruned` conversion that is the
        # UNPRUNED file, which is the one base this artifact will refuse to load
        # on, so it is exactly the wrong thing to read when asking "what does
        # this fit". These two say so directly. Added 2026-08-31; a file older
        # than that carries neither, and the node classifies by key prefix
        # instead, which is the observable and needs no metadata at all.
        "h3_pdd_adaln_form": "baked" if args.pruned is not None else "2688",
        "h3_pdd_loads_on": (
            ("a checkpoint with this backbone BAKED IN at strength "
             f"{baked_strength:g}, built from " if args.omit_backbone else "")
            + ("the pruned/curve-form build of this partition only"
               if args.pruned is not None
               else "either the pruned or the unpruned build")),
        "h3_pdd_pruned_base": (args.pruned.name if args.pruned is not None
                               else ""),
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
        # What the FILE carries, counted as modules including the refiner
        # (208 full, 8 stripped); `backbone_modules_converted` is the number
        # converted regardless. Neither is a tensor count: 3 tensors each.
        "backbone_modules": str(modules_in_file),
        "backbone_modules_converted": str(modules),
        "h3_pdd_backbone": "stripped" if args.omit_backbone else "full",
        "h3_pdd_backbone_strength_baked": (repr(baked_strength)
                                           if args.omit_backbone else ""),
        "h3_pdd_backbone_probe_key": BACKBONE_PROBE_KEY,
        "h3_pdd_backbone_probe_rows": str(BACKBONE_PROBE_ROWS),
        "h3_pdd_backbone_probe_of": "baked" if args.omit_backbone else "base",
        "h3_pdd_backbone_probe_from": probe_src.name,
        "h3_pdd_backbone_probe_sha256": probe_digest(probe_codes, probe_scale),
        "h3_pdd_baked_checkpoint": (args.baked.name if args.omit_backbone
                                    else ""),
        "adaln_modules": str(adaln_modules),
        "source_format": "alibaba-pai PDD (diffusers naming, stacked heads)",
        "target_format": "ComfyUI generic LoRA + h3_pdd.* sidecar tensors",
        "swi_glu_mapping": "release [value;gate] -> ComfyUI [gate;value]",
        "qkv_fusion": "concat A; block-diagonal B; alpha multiplied by 3",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out), metadata=metadata)

    print(f"wrote {args.out}")
    print(f"  {modules} backbone modules converted, {adaln_modules} adaln "
          f"modules ({len(src)} source tensors, all consumed)")
    if args.omit_backbone:
        print(f"  STRIPPED: {stripped} backbone tensors dropped, "
              f"{modules_in_file} modules left under diffusion_model.* (the "
              f"refiner); baked strength {baked_strength:g}")
    print(f"  backbone probe {BACKBONE_PROBE_KEY}[:{BACKBONE_PROBE_ROWS}] "
          f"codes+scale from {probe_src.name} "
          f"({metadata['h3_pdd_backbone_probe_of']}), digest "
          f"{metadata['h3_pdd_backbone_probe_sha256'][:16]}; population "
          f"{pop['blocks']}x{BACKBONE_KINDS_PER_BLOCK} + "
          f"{pop['refiner_blocks']}x{BACKBONE_KINDS_PER_BLOCK} asserted "
          f"against the release")
    print(f"  {out['h3_pdd.bank.video.weight'].shape[0]}-interval head bank at "
          f"shift {args.shift_video}/{args.shift_audio}")
    print("  bank rows are verbatim heads, not deltas: "
          + ", ".join(f"{k.split('.')[0]} {r:.4f}"
                      for k, r in sorted(bank_ratios.items()))
          + f" (delta would be ~0.02, threshold {BANK_VERBATIM_RATIO})")
    # The grid is emitted ONLY on the unpruned path, so reading it
    # unconditionally raised KeyError after the file had already been written --
    # every `--pruned` run looked failed and returned non-zero while having
    # produced a correct artifact. Both files this repo ships are pruned
    # conversions, so that was every real run.
    if "h3_pdd.silu_temb_grid" in out:
        print(f"  time grid {tuple(out['h3_pdd.silu_temb_grid'].shape)} "
              f"from {args.base.name}")
    else:
        print(f"  adaln baked into {baked} block(s) of the pruned curve basis; "
              f"no runtime grid needed")
    print(f"  partition fingerprint {metadata['base_video_out_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
