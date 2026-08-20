#!/usr/bin/env python3
"""What is inside the fl2va / ref2va checkpoints, measured, not inferred.

Streams tensors out of the two int8_convrot checkpoints by safetensors
header offsets (no GPU, no safetensors dependency), dequantizes the int8
linears against their per-channel scales, and measures per block: relative
Frobenius delta and cosine for the four quantized linears, the norms, and
the AdaLN projection -- the last one at the MODULATION OUTPUT, not at the
stored coefficients, for the reason below; plus the global tensors and the
per-block distribution of modulation delta that the hybrid checkpoints
slice.

## What the 2026-08-18 run got wrong, and why this script changed

The first version of this file (record: bench/results/2026-08-18_dit_internals.json)
compared `blocks.N.adaln_proj.linear.weight` between the checkpoints
directly and reported "AdaLN is replaced, not adjusted, in every block:
rel-delta ~1.9 with negative cosine", plus a per-block "delta energy"
profile it described as flat. Both statements are artifacts of the
checkpoint format, found 2026-08-20.

These are curve-form checkpoints: the full-width adaln weight is replaced by
an 8-column basis of the time-embedding curve (`adaln_t_table`, [1025, 8])
and a per-block coefficient matrix ([96768, 8]). What the block consumes is
their product, `table[t] @ W.T + b`. The two checkpoints were factorised
separately, and their bases agree up to SIGN on only half the columns:
per-column cosine of the two tables is +1.0, +1.0, +0.996, +0.996, -0.9997,
-0.9997, -0.99, -0.99 (the `adaln_t_table_column_cos` field). The
coefficient columns that multiply the flipped basis columns are the
large-norm ones, so a direct coefficient comparison returns a large,
negatively-signed delta for a modulation that is nearly the same. The
"energy" profile was the squared norm of that same sign flip.

The stored coefficients are still reported, under `basis_dependent`, so
the retraction stays legible against the old record; nothing should be
quoted from them.

## What the modulation-level comparison says (2026-08-20 run)

  - **The modulation differs by a few percent, like everything else.**
    `mod_rel_delta` (the whole output over the time grid) runs 1.5-4.7% per
    block, tracking the bias delta almost exactly because the bias is
    90-95% of the modulation's norm. The time-varying part alone
    (`mod_t_rel_delta`) differs 5-9% per block, with cosine ~1. The final
    layer's time-varying part is the largest single item at ~12%. Nothing
    is replaced; nothing has a negative cosine.
  - **So the hybrid checkpoints swap a few-percent modulation difference,
    not "the reference pathway".** Their README premise -- that the
    checkpoints' differences are concentrated in `adaln_proj` -- is the
    same coefficient-level comparison on the same curve-form files. The
    linears they keep from fl2va differ from ref2va's by the same ~3%.
    The `hybrid_adaln_mod_t_delta_energy_fraction` field is what the
    cutoffs cover in functional terms. A separate check (2026-08-20,
    scratch, not a record) found the hybrids apply ref2va's coefficients
    to fl2va's table; because the flipped basis columns carry ~0.1% of
    the modulation norm, that mis-pairing costs them ~0.2% at the output,
    i.e. nothing.
  - **Block 49's text rows are a no-op in both checkpoints.** Chunks 8-11
    of its modulation output -- `gate_msa`, `shift_mlp`, `scale_mlp`,
    `gate_mlp` for modality tag 1, which `model.py` maps to text -- are
    exactly zero in weight and bias (`zero_chunks`). Both gates zero means
    text tokens pass through the last block unchanged. Not a
    differentiator; recorded because it is the one structurally special
    thing about block 49 at the weight level and the reader will look.
  - **The int8 linears differ ~3.2% relative on average** (per-tensor range
    2.0-3.8%, cosine at or above 0.9993 at every block); the output heads
    differ more (video_out 5.0%, audio_out 7.9%); condition/patch
    projections ~2.2-2.4%. Unchanged from the 2026-08-18 run, reproduced
    exactly, and that reproduction is the port control for this rewrite.
    All rel-deltas are ||ref2va - fl2va|| over ||ref2va||.
  - **Quantization stress is identical** between the two checkpoints
    (qkv per-channel scale mean/max match to 4 decimals per block), so
    neither partition is the harder int8 target.

Chunk layout, from `comfy/ldm/minimax/model.py::AdalnProj.forward`: the
96768-wide output is viewed as [3 modalities, 6 params x 5376], so chunk
index c (of 18, each 5376 wide) is `modality * 6 + param`, params in the
order shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, and
`seg_tag` maps video -> 0, text -> 1, audio -> 2.

The checkpoint directory is resolved from this repo's own location inside
`custom_nodes` (../../models/diffusion_models), overridable with
--models-dir for a checkout that lives elsewhere.

Usage:
    bench/analyze_checkpoint_delta.py out.json [--models-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from datetime import date
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

_DT = {"F32": np.float32, "F16": np.float16, "I8": np.int8, "U8": np.uint8}

FL_NAME = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
RE_NAME = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
HYBRID_CUTS = (15, 20, 25, 30)

INT8_MODS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
GLOBAL_TENSORS = (
    "condition_proj.weight", "video_patch_proj.weight",
    "audio_patch_proj.weight", "final_layer.adaln_proj.linear.weight",
    "final_layer.video_out.weight", "final_layer.audio_out.weight",
    "adaln_t_table")


def header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def read(path: str, hdr: dict, base: int, name: str) -> np.ndarray:
    info = hdr[name]
    o0, o1 = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + o0)
        buf = f.read(o1 - o0)
    dt = info["dtype"]
    if dt == "BF16":
        arr = (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
               ).view(np.float32)
    else:
        arr = np.frombuffer(buf, dtype=_DT[dt]).astype(np.float32)
    return arr.reshape(info["shape"])


def deq(path: str, hdr: dict, base: int, mod: str) -> np.ndarray:
    return (read(path, hdr, base, mod + ".weight")
            * read(path, hdr, base, mod + ".weight_scale"))


def rel_delta(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt((d * d).sum())) / max(float(np.sqrt((a * a).sum())),
                                               1e-12)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum()) / max(
        float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())), 1e-12)


def cos64(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine with float64 accumulation. The modulation matrices are ~1e8
    elements; a float32 reduction returned cosines of 1.004 on the scratch
    run, which cannot ship."""
    a64, b64 = a.astype(np.float64, copy=False), b.astype(np.float64, copy=False)
    return float((a64 * b64).sum() / max(
        np.sqrt((a64 * a64).sum()) * np.sqrt((b64 * b64).sum()), 1e-12))


def rel_delta64(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / ||a||, float64, nan when ||a|| is zero (a zero chunk)."""
    a64, b64 = a.astype(np.float64, copy=False), b.astype(np.float64, copy=False)
    na = float(np.sqrt((a64 * a64).sum()))
    if na == 0.0:
        return float("nan")
    d = a64 - b64
    return float(np.sqrt((d * d).sum())) / na


N_CHUNKS = 18        # 3 modalities x 6 params
CHUNK_W = 5376       # hidden


def modulation(table: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The adaln output over the whole time grid: [grid, out] in float64.
    This is what `AdalnProj` computes for every t the sampler can ask for,
    and the only level at which the two checkpoints are comparable."""
    return table.astype(np.float64) @ w.astype(np.float64).T + b.astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="JSON output path")
    ap.add_argument("--models-dir", default=None,
                    help="ComfyUI diffusion_models dir; default resolves "
                         "../../models/diffusion_models from this repo")
    ap.add_argument("--measured", default=date.today().isoformat(),
                    help="observation date stamped into the record")
    args = ap.parse_args()

    models = Path(args.models_dir) if args.models_dir else (
        Path(__file__).resolve().parents[3] / "models" / "diffusion_models")
    fl = str(models / FL_NAME)
    re_ = str(models / RE_NAME)

    fl_h, fl_b = header(fl)
    re_h, re_b = header(re_)
    out: dict[str, Any] = {
        "measured": args.measured,
        "produced_by": "bench/analyze_checkpoint_delta.py",
        "what": ("fl2va vs ref2va int8_convrot internals: per-block dequantized "
                 "linear deltas, adaln compared at the modulation output over "
                 "the time grid, hybrid coverage of the time-varying modulation "
                 "delta"),
        "files": {"fl2va": FL_NAME, "ref2va": RE_NAME},
        "adaln_comparison_level": (
            "modulation output over the adaln_t_table grid; the stored "
            "coefficient matrices are basis-dependent and reported only "
            "under basis_dependent"),
        "per_block": [], "scales": [], "global": {}}

    tf = read(fl, fl_h, fl_b, "adaln_t_table")
    tr = read(re_, re_h, re_b, "adaln_t_table")
    out["adaln_t_table"] = {
        "shape": list(tf.shape),
        "rel_delta": round(rel_delta(tr, tf), 5),
        "column_cos": [round(cos64(tf[:, j], tr[:, j]), 5) for j in range(tf.shape[1])],
        "fl_column_norms": [round(float(np.linalg.norm(tf[:, j])), 4) for j in range(tf.shape[1])],
    }

    mod_t_e: list[float] = []
    for blk in range(50):
        row: dict[str, Any] = {"block": blk}
        for mod in INT8_MODS:
            name = f"blocks.{blk}.{mod}"
            a = deq(fl, fl_h, fl_b, name)
            b = deq(re_, re_h, re_b, name)
            row[mod] = {"rel_delta": round(rel_delta(b, a), 5),
                        "cos": round(cos(a, b), 6)}
            if mod == "attn.qkv_proj":
                sa = read(fl, fl_h, fl_b, name + ".weight_scale").ravel()
                sb = read(re_, re_h, re_b, name + ".weight_scale").ravel()
                out["scales"].append({
                    "block": blk,
                    "fl_qkv_scale_mean": float(sa.mean()),
                    "fl_qkv_scale_max": float(sa.max()),
                    "re_qkv_scale_mean": float(sb.mean()),
                    "re_qkv_scale_max": float(sb.max())})
            del a, b
        wa = read(fl, fl_h, fl_b, f"blocks.{blk}.adaln_proj.linear.weight")
        wb = read(re_, re_h, re_b, f"blocks.{blk}.adaln_proj.linear.weight")
        ba = read(fl, fl_h, fl_b, f"blocks.{blk}.adaln_proj.linear.bias")
        bb = read(re_, re_h, re_b, f"blocks.{blk}.adaln_proj.linear.bias")
        ma = modulation(tf, wa, ba)
        mb = modulation(tr, wb, bb)
        mta = ma - ma.mean(axis=0)
        mtb = mb - mb.mean(axis=0)
        zero_chunks = [c for c in range(N_CHUNKS)
                       if not np.any(wa[c * CHUNK_W:(c + 1) * CHUNK_W])
                       and not np.any(ba[c * CHUNK_W:(c + 1) * CHUNK_W])
                       and not np.any(wb[c * CHUNK_W:(c + 1) * CHUNK_W])
                       and not np.any(bb[c * CHUNK_W:(c + 1) * CHUNK_W])]
        row["adaln"] = {
            "mod_rel_delta": round(rel_delta64(mb, ma), 5),
            "mod_cos": round(cos64(ma, mb), 6),
            "mod_t_rel_delta": round(rel_delta64(mtb, mta), 5),
            "mod_t_cos": round(cos64(mta, mtb), 6),
            "mod_t_fraction_ref2va": round(float(np.linalg.norm(mtb) / np.linalg.norm(mb)), 5),
            "mod_t_rel_delta_by_chunk": [
                round(rel_delta64(mtb[:, c * CHUNK_W:(c + 1) * CHUNK_W],
                                  mta[:, c * CHUNK_W:(c + 1) * CHUNK_W]), 4)
                for c in range(N_CHUNKS)],
            "zero_chunks": zero_chunks,
            "b_rel_delta": round(rel_delta(bb, ba), 5),
            "basis_dependent": {
                "w_rel_delta": round(rel_delta(wb, wa), 5),
                "w_cos": round(cos(wa, wb), 6),
                "note": "coefficients on two differently-signed bases; not comparable"},
        }
        dd = mtb - mta
        mod_t_e.append(float((dd * dd).sum()))
        del ma, mb, mta, mtb, dd
        norms: dict[str, float] = {}
        for nm in ("norm1", "norm2", "attn.q_norm", "attn.k_norm"):
            na = read(fl, fl_h, fl_b, f"blocks.{blk}.{nm}.weight")
            nb = read(re_, re_h, re_b, f"blocks.{blk}.{nm}.weight")
            norms[nm] = round(rel_delta(nb, na), 5)
        row["norms_rel_delta"] = norms
        out["per_block"].append(row)
        print(f"block {blk} done", flush=True)

    for nm in GLOBAL_TENSORS:
        a = read(fl, fl_h, fl_b, nm)
        b = read(re_, re_h, re_b, nm)
        out["global"][nm] = {"rel_delta": round(rel_delta(b, a), 5),
                             "cos": round(cos(a, b), 6)}
        if nm == "final_layer.adaln_proj.linear.weight":
            out["global"][nm]["basis_dependent"] = True
            ba = read(fl, fl_h, fl_b, "final_layer.adaln_proj.linear.bias")
            bb = read(re_, re_h, re_b, "final_layer.adaln_proj.linear.bias")
            ma, mb = modulation(tf, a, ba), modulation(tr, b, bb)
            mta, mtb = ma - ma.mean(axis=0), mb - mb.mean(axis=0)
            out["global"]["final_layer.adaln_proj (modulation)"] = {
                "mod_rel_delta": round(rel_delta64(mb, ma), 5),
                "mod_cos": round(cos64(ma, mb), 6),
                "mod_t_rel_delta": round(rel_delta64(mtb, mta), 5),
                "mod_t_cos": round(cos64(mta, mtb), 6),
                "mod_t_fraction_ref2va": round(float(np.linalg.norm(mtb) / np.linalg.norm(mb)), 5)}
            del ma, mb, mta, mtb

    # What each hybrid cutoff covers, in the only currency that is
    # functional: the squared norm of the time-varying modulation
    # difference, summed over the blocks it swaps. The 2026-08-18 record's
    # `hybrid_adaln_delta_energy_fraction` was this quantity computed on the
    # sign-flipped coefficients and is withdrawn.
    tot = sum(mod_t_e)
    out["hybrid_adaln_mod_t_delta_energy_fraction"] = {
        f"b{c}-49": round(sum(mod_t_e[c:]) / tot, 4) for c in HYBRID_CUTS}
    out["adaln_mod_t_delta_energy_by_block"] = [round(e / tot, 5) for e in mod_t_e]

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
