#!/usr/bin/env python3
"""What the Comfy-Org "pruned" H3 checkpoints lose, measured per block and per
checkpoint, against the unpruned int8_convrot files.

## What "pruned" is

The unpruned DiT carries a timestep MLP (`time_embedder`, 256 -> 5376 -> 2688,
fp32) and a full-width AdaLN projection per block, `adaln_proj.linear.weight
[96768, 2688]` (13.0 B parameters over the 50 blocks plus the final layer, 41%
of the DiT by count). The pruned file drops the MLP and replaces every
projection with a rank-8 form: `adaln_t_table [1025, 8]` (fp32, one row per
grid point of t in [0, 1]) and per-block coefficients `[96768, 8]` (fp16). The
runtime (`comfy/ldm/minimax/model.py`) lerps the table at the requested t and
multiplies; the unpruned runtime computes `W @ silu(e(t)) + b`. Nothing in
this repo had compared the two before 2026-08-20: every fl2va-vs-ref2va
measurement was pruned against pruned, so a loss introduced by the pruning,
and any asymmetry of that loss between the two checkpoints, was invisible.

## What this script measures

Per checkpoint, on CPU, block by block:

  - `embedder_check`: the numpy replica of `TimeEmbedder` against comfy's own
    module on the same weights. A failed factorisation test below would
    otherwise be ambiguous between wrong Fourier maths and a wrong hypothesis.
  - `curve`: the time-embedding curve `S = silu(e(t))` over the 1025-point
    grid, centred over t; its singular values; the residual of a least-squares
    fit of the centred curve onto the shipped table (`S - mean ~ T @ B`),
    against the best rank-8 residual from the SVD. The first run (2026-08-20,
    scratch) fitted the uncentred curve and read a residual of 0.95: the
    construction centres first. Equal residuals, and table column norms equal
    to the centred curve's singular values, mean the table is `U @ Sigma` of
    the centred curve; `folded_bias_rel_delta` below checks the other half,
    that the mean went into the bias. Together they turn "rank-8
    factorisation" from a shape inference into a measured construction.
  - `per_block`: with `W_full` dequantised and un-rotated by comfy_kitchen's
    own ConvRot routine (group 64 on these tensors), the modulation output of
    both forms over the grid. Three numbers: `mod_rel_delta` (whole output,
    the functional quantity), the headline `mod_t_rel_delta` (each side minus
    its own mean over t, the part the table has to carry; the constant part is
    90-95% of the norm and would hide everything), and
    `folded_bias_rel_delta` (pruned bias against `b_full + W_full @ mean S`,
    the bias the construction implies). Per chunk (18 = 3 modalities x 6
    params, layout in `analyze_checkpoint_delta.py`) the centred residual
    beside the chunk's own norm, so a tiny-denominator artifact is legible.
    `basis_only_rel_delta` projects the centred curve onto the table's span
    and runs it through `W_full`: what the pruning could achieve at best with
    this table, separating basis loss from coefficient fitting and fp16.
  - `at_sampling_timesteps`: the same residual at the t values a render asks
    for: the 16-step base arm (shift 12 video / 3 audio), the 4-step turbo arm
    (shift 6 / 3), their audio-shifted counterparts, and the 0.999 condition
    timestep. Trailing sigma 0 is dropped (never evaluated); the pruned side
    goes through the same lerp the runtime uses.
  - `final_layer`: the same comparison on `final_layer.adaln_proj`, which the
    unpruned file stores in bf16, not int8. That makes it the one int8-free
    measurement of the pruning loss in the file.
  - `shared_tensors`: byte comparison of every non-AdaLN tensor between the
    pruned and unpruned file of the same checkpoint. Identical means the
    pruned file is the unpruned file with the AdaLN swapped, and every residual
    here attributes to that swap alone; different means two quantisation runs
    and a looser attribution. The record says which.
  - `checkpoints`: the embedder curve of fl2va against ref2va, so "did e(t)
    move in the fine-tune, or only the projections" has a number.

## Caveats that travel with the numbers

The unpruned block AdaLN is int8_convrot, so every per-block residual is
"rank-8 fp16 form vs int8 form", an upper bound on the pruning loss that
includes int8 error on the full weight. The final-layer row is the clean one.
Separating the two on the blocks needs the bf16 release. The fl2va-vs-ref2va
comparison of residuals is unaffected: both sides carry the same treatment.

## What the 2026-08-20 run found (record: bench/results/2026-08-20_adaln_pruning_residual.json)

  - The construction, measured: `adaln_t_table` is `U @ Sigma` of the centred
    `silu(e(t))` curve (column norms equal the singular values to four
    digits; fit residual 4.3-4.5e-5, equal to the rank-8 SVD residual), and
    the curve's mean is folded into the bias (`folded_bias_rel_delta` 0.1-0.2%,
    the int8/fp16 floor). Every non-AdaLN tensor is byte-identical between
    the pruned and unpruned file of each checkpoint (829 of 829), so the
    residuals attribute to the AdaLN swap alone.
  - Whole-output residual 0.11-0.24% per block, 0.06-0.35% at the timesteps a
    render evaluates, on both checkpoints. The bf16 final layer reads ~0.02%,
    and the basis alone loses ~0.02%: the per-block figure is mostly int8 on
    the unpruned side, and the rank-8 truncation itself is at the 2e-4 level.
  - fl2va and ref2va lose the same: per-block residual ratio 0.97-1.01. The
    hypothesis this script was written to test, that ref2va compresses worse
    under its own factorisation, is refuted.
  - The time embedder differs between the checkpoints (e(t) 10.6% relative,
    cosine 0.994); the fine-tune moved it, not only the projections.
  - Block 49 chunks 8-11 (text-row params) are exactly zero in both forms,
    as `analyze_checkpoint_delta.py` recorded; their residual is nan by design.

## Self-test (`--self-test`)

Two deliberate violations, run before the measurement and refused if either
passes: the pruned file offered as the unpruned one (must be rejected for
having no time embedder), and a block whose pruned coefficients are zeroed
(its t-varying residual must read ~1). A metric that cannot fail is not
a measurement.

Paths to the model files are resolved from the ComfyUI checkout or typed on
the command line; the record carries filenames only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _paths  # noqa: E402
from analyze_checkpoint_delta import (  # noqa: E402
    CHUNK_W, N_CHUNKS, cos64, header, read, rel_delta64)

_COMFY = _paths._comfy_root()
if _COMFY is None:
    _COMFY = Path(os.environ.get("COMFY_ROOT", ""))
sys.path.insert(0, str(_COMFY))

from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    dequantize_int8_convrot_weight)

PRUNED = {"fl2va": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
          "ref2va": "minimax_h3_ref2va_pruned_int8_convrot.safetensors"}
UNPRUNED = {"fl2va": "minimax_h3_fl2va_int8_convrot.safetensors",
            "ref2va": "minimax_h3_ref2va_int8_convrot.safetensors"}

GRID = 1025
COND_T = 0.999           # VISUAL_COND_TIMESTEP in comfy/ldm/minimax/model.py
ARMS = (                 # (label, steps, video shift, audio shift)
    ("base_16step_shift12", 16, 12.0, 3.0),
    ("turbo_4step_shift6", 4, 6.0, 3.0),
)


# ---------------------------------------------------------------- embedding

def fourier(t: np.ndarray, dim: int = 256) -> np.ndarray:
    half = dim // 2
    freqs = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    args = t.astype(np.float32)[:, None] * freqs[None]
    return np.concatenate([np.cos(args), np.sin(args)], axis=-1)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def time_embed(t: np.ndarray, te: dict) -> np.ndarray:
    """Replica of `TimeEmbedder.forward`: proj_out(silu(proj_in(fourier(t))))."""
    h = fourier(t) @ te["proj_in.weight"].T + te["proj_in.bias"]
    return silu(h) @ te["proj_out.weight"].T + te["proj_out.bias"]


def embedder_weights(path: str, hdr: dict, base: int) -> dict:
    if "time_embedder.proj_in.weight" not in hdr:
        raise ValueError(f"{Path(path).name} has no time_embedder: it is a "
                         "pruned (curve-form) checkpoint, not the unpruned one")
    return {k: read(path, hdr, base, "time_embedder." + k) for k in
            ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias")}


def check_embedder(te: dict) -> dict:
    """numpy replica vs comfy's TimeEmbedder on the same fp32 weights."""
    import comfy.ops
    from comfy.ldm.minimax.model import TimeEmbedder
    mod = TimeEmbedder(256, 5376, 2688, dtype=torch.float32, device="cpu",
                       operations=comfy.ops.disable_weight_init)
    sd = {"proj_in.weight": te["proj_in.weight"], "proj_in.bias": te["proj_in.bias"],
          "proj_out.weight": te["proj_out.weight"], "proj_out.bias": te["proj_out.bias"]}
    mod.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in sd.items()})
    t = np.array([0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0], dtype=np.float32)
    with torch.no_grad():
        ref = mod(torch.from_numpy(t)).numpy()
    mine = time_embed(t, te)
    return {"t": t.tolist(), "max_abs_diff": float(np.abs(ref - mine).max()),
            "rel_l2": rel_delta64(ref, mine)}


def lerp_table(table: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The runtime's table read (`model.py`): fractional grid index, lerp,
    t=1.0 held on the last interval."""
    n = table.shape[0]
    pos = np.clip(t.astype(np.float64), 0.0, 1.0) * (n - 1)
    i0 = np.minimum(np.floor(pos).astype(np.int64), n - 2)
    frac = (pos - i0)[:, None]
    return table[i0].astype(np.float64) * (1 - frac) + table[i0 + 1].astype(np.float64) * frac


# ---------------------------------------------------------------- timesteps

def sampling_timesteps() -> dict:
    """t values the model is evaluated at, per arm: video t, audio t."""
    import comfy.cli_args
    comfy.cli_args.args.cpu = True          # model_management must not touch the card
    import comfy.model_sampling
    import comfy.samplers
    from comfy.ldm.minimax.model import time_shift_sigma
    out = {}
    for label, steps, shift, audio_shift in ARMS:
        ms = comfy.model_sampling.ModelSamplingAV(None)
        ms.set_parameters(shift=shift, audio_shift=audio_shift)
        sig = comfy.samplers.simple_scheduler(ms, steps)[:-1].double()
        sig = sig.clamp(min=1e-6)
        t_v = (1.0 - sig).numpy()
        t_a = (1.0 - time_shift_sigma(sig, shift, audio_shift)).numpy()
        out[label] = {"steps": steps, "shift": shift, "audio_shift": audio_shift,
                      "video_t": [round(float(x), 6) for x in t_v],
                      "audio_t": [round(float(x), 6) for x in t_a]}
    out["condition"] = {"video_t": [COND_T], "audio_t": [1.0]}
    return out


def flat_timesteps(ts: dict) -> tuple[np.ndarray, list[tuple[str, str, int]]]:
    vals, tags = [], []
    for arm, d in ts.items():
        for kind in ("video_t", "audio_t"):
            for i, v in enumerate(d[kind]):
                vals.append(v)
                tags.append((arm, kind, i))
    return np.array(vals, dtype=np.float64), tags


# ---------------------------------------------------------------- per block

def full_weight(path: str, hdr: dict, base: int, mod: str) -> tuple[np.ndarray, np.ndarray]:
    """Dequantised, un-rotated fp32 weight and fp32 bias of an unpruned adaln."""
    info = hdr[mod + ".weight"]
    if info["dtype"] == "I8":
        q = torch.from_numpy(read(path, hdr, base, mod + ".weight").astype(np.int8))
        scale = torch.from_numpy(read(path, hdr, base, mod + ".weight_scale"))
        cq = hdr.get(mod + ".comfy_quant")
        if cq is None:
            raise ValueError(f"{mod}: int8 weight without a comfy_quant marker")
        o0, o1 = cq["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + o0)
            conf = json.loads(f.read(o1 - o0).decode())
        if not conf.get("convrot"):
            raise ValueError(f"{mod}: expected a convrot int8 tensor, got {conf}")
        w = dequantize_int8_convrot_weight(q, scale, int(conf["convrot_groupsize"])).numpy()
    else:
        w = read(path, hdr, base, mod + ".weight")
    return w, read(path, hdr, base, mod + ".bias")


def compare(s_grid: np.ndarray, s_proj: np.ndarray, t_grid8: np.ndarray,
            s_ts: np.ndarray, t_ts8: np.ndarray, ts_tags,
            w_full: np.ndarray, b_full: np.ndarray,
            w8: np.ndarray, b8: np.ndarray) -> dict:
    """All residuals for one projection. Matmuls in fp32 torch (CPU threads),
    reductions in fp64."""
    wf = torch.from_numpy(np.ascontiguousarray(w_full, dtype=np.float32))
    w8t = torch.from_numpy(np.ascontiguousarray(w8, dtype=np.float32))

    def mm(s, w):
        return (torch.from_numpy(np.ascontiguousarray(s, dtype=np.float32)) @ w.T).numpy()

    m_full = mm(s_grid, wf) + b_full[None]          # whole output, full form
    m_proj = mm(s_proj, wf) + b_full[None]          # full weights on the table's span
    m_pr = mm(t_grid8, w8t) + b8[None]              # whole output, pruned form
    mt_full = m_full - m_full.mean(axis=0)          # centred over t: the table's job
    mt_proj = m_proj - m_proj.mean(axis=0)
    mt_pr = m_pr - m_pr.mean(axis=0)
    folded = b_full.astype(np.float64) + wf.double().numpy() @ s_grid.astype(np.float64).mean(axis=0)

    out: dict = {
        "folded_bias_rel_delta": rel_delta64(folded, b8),
        "raw_bias_rel_delta": rel_delta64(b_full, b8),
        "mod_norm": float(np.linalg.norm(m_full.astype(np.float64))),
        "mod_rel_delta": rel_delta64(m_full, m_pr),
        "mod_t_rel_delta": rel_delta64(mt_full, mt_pr),
        "mod_t_cos": cos64(mt_full, mt_pr),
        "mod_t_norm": float(np.linalg.norm(mt_full.astype(np.float64))),
        "basis_only_rel_delta": rel_delta64(mt_full, mt_proj),
    }
    chunks = []
    for c in range(N_CHUNKS):
        sl = slice(c * CHUNK_W, (c + 1) * CHUNK_W)
        a, b = mt_full[:, sl], mt_pr[:, sl]
        chunks.append({"chunk": c,
                       "mod_t_norm": float(np.linalg.norm(a.astype(np.float64))),
                       "mod_t_rel_delta": rel_delta64(a, b)})
    out["chunks"] = chunks

    # the t values a render evaluates
    f_ts = mm(s_ts, wf) + b_full[None]
    p_ts = mm(t_ts8, w8t) + b8[None]
    mu_f, mu_p = m_full.mean(axis=0), m_pr.mean(axis=0)
    rows = []
    for i, (arm, kind, step) in enumerate(ts_tags):
        rows.append({"arm": arm, "kind": kind, "step": step,
                     "mod_rel_delta": rel_delta64(f_ts[i], p_ts[i]),
                     "mod_t_rel_delta": rel_delta64(f_ts[i] - mu_f, p_ts[i] - mu_p)})
    out["at_sampling_timesteps"] = rows
    return out


def shared_tensor_compare(pruned: str, p_h: dict, p_b: int,
                          unpruned: str, u_h: dict, u_b: int) -> dict:
    """Byte-compare every tensor the two files share, AdaLN and table excluded."""
    same, diff, missing = [], [], []
    for k in p_h:
        if "adaln" in k:
            continue
        if k not in u_h:
            missing.append(k)
            continue
        if p_h[k]["dtype"] != u_h[k]["dtype"] or p_h[k]["shape"] != u_h[k]["shape"]:
            diff.append(k)
            continue
        o0, o1 = p_h[k]["data_offsets"]
        q0, q1 = u_h[k]["data_offsets"]
        with open(pruned, "rb") as f, open(unpruned, "rb") as g:
            f.seek(p_b + o0)
            g.seek(u_b + q0)
            eq = f.read(o1 - o0) == g.read(q1 - q0)
        (same if eq else diff).append(k)
    return {"compared": len(same) + len(diff), "identical": len(same),
            "different": len(diff), "missing_in_unpruned": missing,
            "different_keys_sample": diff[:12],
            "unpruned_only": sorted(k for k in u_h if k not in p_h and "adaln" not in k
                                    and not k.startswith("time_embedder"))}


# ---------------------------------------------------------------- main

def analyse_checkpoint(tag: str, pruned: str, unpruned: str, blocks: list[int],
                       ts: dict, skip_shared: bool) -> dict:
    p_h, p_b = header(pruned)
    u_h, u_b = header(unpruned)
    te = embedder_weights(unpruned, u_h, u_b)
    rec = {"pruned_file": Path(pruned).name, "unpruned_file": Path(unpruned).name,
           "embedder_check": check_embedder(te)}

    grid_t = np.linspace(0.0, 1.0, GRID, dtype=np.float64)
    e_grid = time_embed(grid_t, te)
    s_grid = silu(e_grid)                                   # [1025, 2688]
    table = read(pruned, p_h, p_b, "adaln_t_table")          # [1025, 8]

    # factorisation check, on the centred curve
    s64 = s_grid.astype(np.float64)
    s_mean = s64.mean(axis=0)
    sc = s64 - s_mean
    t64 = table.astype(np.float64)
    coef, *_ = np.linalg.lstsq(t64, sc, rcond=None)          # B: [8, 2688]
    s_proj = t64 @ coef + s_mean                             # best the table can do
    sv = np.linalg.svd(sc, compute_uv=False)
    energy = sv ** 2
    rec["curve"] = {
        "grid_points": GRID,
        "centred": True,
        "mean_norm": float(np.linalg.norm(s_mean)),
        "singular_values_top12": [round(float(x), 5) for x in sv[:12]],
        "table_column_norms": [round(float(x), 5) for x in np.linalg.norm(t64, axis=0)],
        "rank8_energy_fraction": float(energy[:8].sum() / energy.sum()),
        "svd_rank8_rel_residual": float(math.sqrt(energy[8:].sum() / energy.sum())),
        "table_fit_rel_residual": rel_delta64(sc, t64 @ coef),
        "uncentred_table_fit_rel_residual": rel_delta64(
            s64, t64 @ np.linalg.lstsq(t64, s64, rcond=None)[0]),
    }

    ts_t, ts_tags = flat_timesteps(ts)
    s_ts = silu(time_embed(ts_t, te))
    t_ts8 = lerp_table(table, ts_t)
    t_grid8 = lerp_table(table, grid_t)

    per_block = []
    for b in blocks:
        mod = f"blocks.{b}.adaln_proj.linear"
        wf, bf = full_weight(unpruned, u_h, u_b, mod)
        w8 = read(pruned, p_h, p_b, mod + ".weight")
        b8 = read(pruned, p_h, p_b, mod + ".bias")
        r = compare(s_grid, s_proj, t_grid8, s_ts, t_ts8, ts_tags, wf, bf, w8, b8)
        r["block"] = b
        r["full_weight_dtype"] = u_h[mod + ".weight"]["dtype"]
        per_block.append(r)
        print(f"  {tag} block {b:2d}: mod {r['mod_rel_delta']:.5f} mod_t {r['mod_t_rel_delta']:.5f} "
              f"basis_only {r['basis_only_rel_delta']:.5f} folded_bias {r['folded_bias_rel_delta']:.5f}",
              flush=True)
        del wf
    rec["per_block"] = per_block

    mod = "final_layer.adaln_proj.linear"
    wf, bf = full_weight(unpruned, u_h, u_b, mod)
    w8 = read(pruned, p_h, p_b, mod + ".weight")
    b8 = read(pruned, p_h, p_b, mod + ".bias")
    n_chunks_final = w8.shape[0] // CHUNK_W
    r = compare(s_grid, s_proj, t_grid8, s_ts, t_ts8, ts_tags, wf, bf, w8, b8)
    r["chunks"] = r["chunks"][:n_chunks_final]
    r["full_weight_dtype"] = u_h[mod + ".weight"]["dtype"]
    rec["final_layer"] = r
    print(f"  {tag} final_layer ({r['full_weight_dtype']}): mod {r['mod_rel_delta']:.5f} "
          f"mod_t {r['mod_t_rel_delta']:.5f} basis_only {r['basis_only_rel_delta']:.5f}", flush=True)

    if not skip_shared:
        rec["shared_tensors"] = shared_tensor_compare(pruned, p_h, p_b, unpruned, u_h, u_b)
        print(f"  {tag} shared tensors: {rec['shared_tensors']['identical']} identical, "
              f"{rec['shared_tensors']['different']} different", flush=True)
    rec["_curve_arrays"] = (e_grid, s_grid)
    return rec


def self_test(pruned: str, unpruned: str, ts: dict) -> dict:
    """Two violations that must be refused."""
    results = {}
    p_h, p_b = header(pruned)
    try:
        embedder_weights(pruned, p_h, p_b)
        results["pruned_as_unpruned"] = "ACCEPTED (violation not caught)"
    except ValueError as e:
        results["pruned_as_unpruned"] = f"refused: {e}"

    u_h, u_b = header(unpruned)
    te = embedder_weights(unpruned, u_h, u_b)
    grid_t = np.linspace(0.0, 1.0, GRID)
    s_grid = silu(time_embed(grid_t, te))
    table = read(pruned, p_h, p_b, "adaln_t_table")
    t_grid8 = lerp_table(table, grid_t)
    ts_t, ts_tags = flat_timesteps(ts)
    s_ts = silu(time_embed(ts_t, te))
    t_ts8 = lerp_table(table, ts_t)
    mod = "blocks.0.adaln_proj.linear"
    wf, bf = full_weight(unpruned, u_h, u_b, mod)
    w8 = read(pruned, p_h, p_b, mod + ".weight")
    b8 = read(pruned, p_h, p_b, mod + ".bias")
    r = compare(s_grid, s_grid, t_grid8, s_ts, t_ts8, ts_tags, wf, bf,
                np.zeros_like(w8), b8)
    results["zeroed_coefficients_block0"] = {
        "mod_t_rel_delta": r["mod_t_rel_delta"],
        "verdict": "red as required" if r["mod_t_rel_delta"] > 0.99 else
                   "GREEN ON A ZEROED BLOCK (metric cannot fail)"}
    ok = results["pruned_as_unpruned"].startswith("refused") and \
        r["mod_t_rel_delta"] > 0.99
    results["passed"] = ok
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="JSON record path")
    ap.add_argument("--models-dir", default=None,
                    help="dir holding the pruned files; default ../../models/diffusion_models")
    ap.add_argument("--unpruned-dir", default=None,
                    help="dir holding the unpruned int8_convrot files; default --models-dir")
    ap.add_argument("--blocks", default=None,
                    help="comma-separated block indices (default all 50)")
    ap.add_argument("--checkpoints", default="fl2va,ref2va")
    ap.add_argument("--skip-shared", action="store_true",
                    help="skip the byte comparison of shared tensors (reads ~20 GB per checkpoint)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the two deliberate violations first; refuse to measure if either passes")
    ap.add_argument("--measured", default=date.today().isoformat())
    args = ap.parse_args()

    models = Path(args.models_dir) if args.models_dir else (
        _HERE.parents[2] / "models" / "diffusion_models")
    unpruned_dir = Path(args.unpruned_dir) if args.unpruned_dir else models
    blocks = ([int(x) for x in args.blocks.split(",")] if args.blocks else list(range(50)))
    tags = [t.strip() for t in args.checkpoints.split(",")]

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    ts = sampling_timesteps()

    out = {
        "measured": args.measured,
        "produced_by": "bench/analyze_adaln_pruning.py",
        "what": ("AdaLN rank-8 pruning residual: Comfy-Org pruned (curve-table) "
                 "checkpoints against the unpruned int8_convrot files, at the "
                 "modulation output, per block and per checkpoint"),
        "caveat": ("block AdaLN in the unpruned file is int8_convrot (group 64), "
                   "so per-block residuals bound the pruning loss and include int8 "
                   "error on the full weight; final_layer is bf16 in the unpruned "
                   "file and is the int8-free measurement"),
        "timesteps": ts,
        "blocks": blocks,
        "checkpoints": {},
    }

    if args.self_test:
        st = self_test(str(models / PRUNED[tags[0]]), str(unpruned_dir / UNPRUNED[tags[0]]), ts)
        out["self_test"] = st
        print("self-test:", json.dumps(st, indent=1))
        if not st["passed"]:
            print("self-test failed; not measuring", file=sys.stderr)
            return 2

    curves = {}
    for tag in tags:
        print(f"== {tag}", flush=True)
        rec = analyse_checkpoint(tag, str(models / PRUNED[tag]),
                                 str(unpruned_dir / UNPRUNED[tag]), blocks, ts,
                                 args.skip_shared)
        curves[tag] = rec.pop("_curve_arrays")
        out["checkpoints"][tag] = rec

    if len(curves) == 2:
        (ea, sa), (eb, sb) = curves[tags[0]], curves[tags[1]]
        out["embedder_between_checkpoints"] = {
            "pair": tags,
            "e_t_rel_delta": rel_delta64(ea, eb), "e_t_cos": cos64(ea, eb),
            "silu_e_t_rel_delta": rel_delta64(sa, sb),
        }
        ra = {r["block"]: r["mod_t_rel_delta"] for r in out["checkpoints"][tags[0]]["per_block"]}
        rb = {r["block"]: r["mod_t_rel_delta"] for r in out["checkpoints"][tags[1]]["per_block"]}
        out["residual_ratio_by_block"] = {
            "what": f"mod_t_rel_delta {tags[1]} / {tags[0]} per block",
            "ratio": [round(rb[b] / ra[b], 4) if ra[b] else None for b in blocks],
        }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
