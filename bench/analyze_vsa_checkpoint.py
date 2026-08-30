#!/usr/bin/env python3
"""What is inside the FastVideo VSA-distilled H3 checkpoint, and how far it moved.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). Needs no server and
no GPU. It streams safetensors a tensor at a time and never holds a whole
checkpoint; the quantized comparison is chunked by rows.

**Not a check.** It asserts nothing and grades nothing. It answers the questions
`docs/research/vsa/fastvideo_vsa_checkpoint.md` is written from.

## The two traps this script exists to route around

**A raw diff of the int8 payload is not a weight distance.** `int8_convrot`
stores `round(W @ H^T / s)` with a per-row fp32 `s`, and the two files do not
use the same rule for `s` -- measured, section `quant_scale_rule`: every row of
the base saturates at |q| = 127, and only about seven in ten rows of the VSA
file do. So the payloads differ even where the underlying weight does not, and
the int8 bytes cannot be compared. What *can* be compared is `q * s`, the
dequantized weight in the rotated basis: `H` is a fixed orthogonal Hadamard of
order 256 (`docs/research/comfyui_h3_t2va_trace.md` section 1.7), identical in
both files because both carry the same `comfy_quant` descriptor, and an
orthogonal transform preserves Frobenius distance exactly. So the relative
distance measured in the rotated basis IS the relative distance in the weight
basis -- but it carries both files' rounding error, which is why every
quantized row here is reported beside the floor that error predicts.

**The pruned adaln factors are only defined up to a basis change.** The pruned
model stores one shared `adaln_t_table` of shape [1025, 8] and a per-block
`[out, 8]` linear, and the model lerps the table by t and applies the linear
(`comfy/ldm/minimax/model.py`, the `adaln_t_table` branch of the forward). Any
invertible 8x8 `V` applied to the table and undone in every block leaves the
model identical, so comparing the factors tensor by tensor reports a large
distance for two identical adaln surfaces. This script compares the **affine map
itself**, exactly, through 9x9 Gram matrices, and verifies that identity against
a materialized product before believing it (`gram_identity_control`).

Usage:

    python bench/analyze_vsa_checkpoint.py
    python bench/analyze_vsa_checkpoint.py --out bench/results/DATE_fastvideo_vsa_checkpoint.json
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open

COMFY = Path.home() / "ComfyUI"
DIT = COMFY / "models" / "diffusion_models"

VSA = DIT / "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors"
#: The pruned files are the right comparison: the VSA checkpoint is itself in the
#: pruned representation, so the unpruned pair would report the pruning as a
#: difference. Both partitions are compared because the key sets are identical
#: and only the numbers say which partition a file was built on.
FL2VA_PRUNED = DIT / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA_PRUNED = DIT / "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
#: The control for "what do two files look like when nothing moved": Comfy-Org's
#: own pruned and unpruned fl2va, which differ by the adaln pruning alone.
FL2VA_UNPRUNED = DIT / "minimax_h3_fl2va_int8_convrot.safetensors"

#: Blocks sampled for the expensive per-tensor work. Front, middle and last,
#: because the one structural pattern this repo has found in H3 backbone deltas
#: runs with depth (`docs/research/h3_partition_distance.md`).
SAMPLE_BLOCKS = (0, 25, 49)
SAMPLE_LINEARS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")

#: Unquantized tensors compared in full. Chosen to cover every stage the packed
#: sequence passes through -- ingest, text refiner, block norms, output heads --
#: so "the trunk did not move" is a claim about the whole path rather than a
#: sample of it. The adaln surface is deliberately absent: it needs the affine
#: comparison, not this one.
FINGERPRINT_KEYS = (
    "video_patch_proj.weight", "video_patch_proj.bias",
    "audio_patch_proj.weight", "audio_patch_proj.bias",
    "condition_proj.weight", "condition_proj.bias",
    "final_layer.video_out.weight", "final_layer.audio_out.weight",
    "final_layer.norm.weight",
    "token_refiner.blocks.0.attn.qkv_proj.weight",
    "token_refiner.blocks.0.attn.out_proj.weight",
    "token_refiner.blocks.1.mlp.fc1.weight",
    "token_refiner.blocks.1.mlp.fc2.weight",
    "token_refiner.final_norm.weight",
    "rope.inv_freq",
)

ROW_CHUNK = 2048


def raw_header(path: Path) -> tuple[dict, int]:
    """The safetensors header as JSON, plus the byte offset the payload starts at."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    head.pop("__metadata__", None)
    return head, 8 + n


def file_metadata(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__")


def collapse(key: str) -> str:
    return re.sub(r"\b\d+\b", "N", key)


def distance(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Relative Frobenius distance and cosine, float64, plus exact-equality.

    `identical` is the load-bearing field. Bit equality is the only statement
    here that needs no error model at all, and it is what separates "unchanged"
    from "changed by an amount too small to matter" -- two readings a distance
    alone cannot tell apart.
    """
    identical = a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))
    a = a.to(torch.float64).flatten()
    b = b.to(torch.float64).flatten()
    nb = float(b.norm())
    na = float(a.norm())
    return {
        "relative": float((a - b).norm() / nb) if nb else None,
        "cosine": float((a @ b) / (na * nb)) if nb and na else None,
        "identical": identical,
        "n_elements_differing": int((a != b).sum()),
        "n_elements": a.numel(),
    }


# --------------------------------------------------------------------------
# 1. inventory


def inventory(path: Path) -> dict:
    head, _ = raw_header(path)
    dtypes, params_by_dtype = Counter(), Counter()
    for spec in head.values():
        n = 1
        for d in spec["shape"]:
            n *= d
        dtypes[spec["dtype"]] += 1
        params_by_dtype[spec["dtype"]] += n
    gates = sorted(k for k in head if "to_gate_compress" in k)
    return {
        "file": path.name,
        "bytes_on_disk": path.stat().st_size,
        "tensors": len(head),
        "parameters": sum(params_by_dtype.values()),
        "tensors_by_dtype": dict(sorted(dtypes.items())),
        "parameters_by_dtype": dict(sorted(params_by_dtype.items())),
        "safetensors_metadata": file_metadata(path),
        "top_level_prefixes": dict(sorted(Counter(k.split(".")[0] for k in head).items())),
        "gate_key_patterns": dict(sorted(Counter(collapse(k) for k in gates).items())),
        "gate_modules": sorted({k.rsplit(".", 1)[0] for k in gates}),
    }


def keyset_diff(a: Path, b: Path) -> dict:
    ha, _ = raw_header(a)
    hb, _ = raw_header(b)
    ka, kb = set(ha), set(hb)
    changed = {}
    for k in sorted(ka & kb):
        if ha[k]["shape"] != hb[k]["shape"] or ha[k]["dtype"] != hb[k]["dtype"]:
            changed[k] = {"base": [hb[k]["dtype"], hb[k]["shape"]],
                          "vsa": [ha[k]["dtype"], ha[k]["shape"]]}
    return {
        "left": a.name,
        "right": b.name,
        "only_in_left": dict(sorted(Counter(collapse(k) for k in ka - kb).items())),
        "only_in_right": dict(sorted(Counter(collapse(k) for k in kb - ka).items())),
        "shared": len(ka & kb),
        "shape_or_dtype_changed": dict(sorted(Counter(collapse(k) for k in changed).items())),
        "shape_or_dtype_changed_example": dict(list(changed.items())[:1]),
    }


# --------------------------------------------------------------------------
# 2. byte identity census
#
# The cheapest statement available and the one that needs no error model. A
# prefix rather than the whole payload keeps this at a few GiB of reads instead
# of forty; a tensor that differs anywhere in its first MiB is reported as
# differing, and one that matches over its prefix is reported as matching over
# its prefix -- never as identical, which is a claim the sample cannot make.

PREFIX_BYTES = 1 << 20


def byte_census(a: Path, b: Path) -> dict:
    ha, oa = raw_header(a)
    hb, ob = raw_header(b)
    same, diff, incomparable = [], [], []
    with open(a, "rb") as fa, open(b, "rb") as fb:
        for k in sorted(set(ha) & set(hb)):
            da, db = ha[k]["data_offsets"], hb[k]["data_offsets"]
            if ha[k]["dtype"] != hb[k]["dtype"] or ha[k]["shape"] != hb[k]["shape"]:
                incomparable.append(k)
                continue
            n = min(da[1] - da[0], PREFIX_BYTES)
            fa.seek(oa + da[0])
            fb.seek(ob + db[0])
            (same if fa.read(n) == fb.read(n) else diff).append(k)
    return {
        "left": a.name,
        "right": b.name,
        "prefix_bytes_compared_per_tensor": PREFIX_BYTES,
        "n_matching_prefix": len(same),
        "n_differing": len(diff),
        "n_incomparable_shape_or_dtype": len(incomparable),
        "matching_prefix_patterns": dict(sorted(Counter(collapse(k) for k in same).items())),
        "differing_patterns": dict(sorted(Counter(collapse(k) for k in diff).items())),
    }


# --------------------------------------------------------------------------
# 3. unquantized fingerprint, against both partitions


def fingerprint(vsa: Path, bases: dict[str, Path]) -> dict:
    out = {}
    handles = {name: safe_open(p, framework="pt") for name, p in bases.items()}
    try:
        with safe_open(vsa, framework="pt") as fv:
            present = set(fv.keys())
            for key in FINGERPRINT_KEYS:
                if key not in present:
                    out[key] = {"absent_from_vsa": True}
                    continue
                a = fv.get_tensor(key)
                out[key] = {name: distance(a, h.get_tensor(key)) for name, h in handles.items()}
            # every block norm, which is cheap and is where bit equality is densest
            norms = {}
            for i in range(50):
                for stem in ("norm1.weight", "norm2.weight",
                             "attn.q_norm.weight", "attn.k_norm.weight"):
                    k = f"blocks.{i}.{stem}"
                    a = fv.get_tensor(k)
                    norms[k] = {name: distance(a, h.get_tensor(k)) for name, h in handles.items()}
            out["_block_norms"] = norms
    finally:
        for h in handles.values():
            h.__exit__(None, None, None)
    return out


def representation_floor(base: Path) -> dict:
    """What a bf16 round trip costs on the fp32 tensors, measured on this file.

    The fp32 fingerprint tensors are the ones where the VSA file differs most
    from the base, and "that is the size of a precision round trip" is the
    reading. Asserting it from the fp32/bf16 mantissa widths would be arithmetic
    about a format; this measures it on the exact tensors being read, so the
    comparison is against a number from the same population.
    """
    out = {}
    with safe_open(base, framework="pt") as fb:
        for key in FINGERPRINT_KEYS:
            t = fb.get_tensor(key)
            if t.dtype is not torch.float32:
                continue
            rt = t.to(torch.bfloat16).to(torch.float32)
            out[key] = distance(rt, t)["relative"]
    return out


def summarise_norms(norms: dict, base_names) -> dict:
    out = {}
    for name in base_names:
        rows = [v[name] for v in norms.values()]
        rels = sorted(r["relative"] for r in rows)
        out[name] = {
            "n": len(rows),
            "n_bit_identical": sum(1 for r in rows if r["identical"]),
            "relative": {"min": rels[0], "median": rels[len(rels) // 2], "max": rels[-1]},
            "max_elements_differing": max(r["n_elements_differing"] for r in rows),
        }
    return out


# --------------------------------------------------------------------------
# 4. the adaln affine map, compared in a basis-invariant way
#
# The stored adaln is `t -> lerp(table)[t] @ W.T + b`. Writing T' = [T | 1]
# ([1025, 9]) and W' = [W | b] ([out, 9]) makes it the single product T' W'.T,
# so the squared distance between two of them expands into traces of 9x9 Gram
# matrices and never materializes the [1025, out] result:
#
#   ||T'v W'v.T - T'b W'b.T||^2
#       = tr((T'v.T T'v)(W'v.T W'v)) + tr((T'b.T T'b)(W'b.T W'b))
#         - 2 tr((T'v.T T'b)(W'b.T W'v))


def _augment(table: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    table = table.to(torch.float64)
    weight = weight.to(torch.float64)
    bias = bias.to(torch.float64)
    tp = torch.cat([table, torch.ones(table.shape[0], 1, dtype=torch.float64)], dim=1)
    wp = torch.cat([weight, bias.unsqueeze(1)], dim=1)
    return tp, wp


def _affine_distance(tv, wv, tb, wb) -> dict:
    sq_v = float(torch.trace((tv.T @ tv) @ (wv.T @ wv)))
    sq_b = float(torch.trace((tb.T @ tb) @ (wb.T @ wb)))
    cross = float(torch.trace((tv.T @ tb) @ (wb.T @ wv)))
    sq_d = max(sq_v + sq_b - 2 * cross, 0.0)
    # How much an elementwise perturbation of the stored factors is magnified in
    # the surface they multiply out to. ||T (dW).T||_F <= ||T||_F ||dW||_F, so a
    # relative perturbation eps of W shows up in the product amplified by at most
    # this ratio. It prices the question the raw distance cannot answer: whether
    # a surface distance of order 1e-2 is a real change or is what storing an
    # 8-column factor in fp16 costs.
    amp = (float(tb.norm()) * float(wb.norm())) / (sq_b ** 0.5) if sq_b else None
    return {
        "relative": (sq_d ** 0.5) / (sq_b ** 0.5) if sq_b else None,
        "cosine": cross / ((sq_v ** 0.5) * (sq_b ** 0.5)) if sq_v and sq_b else None,
        "frobenius_norm_vsa": sq_v ** 0.5,
        "frobenius_norm_base": sq_b ** 0.5,
        "factor_to_product_amplification": amp,
    }


def adaln_affine(vsa: Path, bases: dict[str, Path]) -> dict:
    modules = [f"blocks.{i}.adaln_proj.linear" for i in range(50)] + ["final_layer.adaln_proj.linear"]
    out = {}
    handles = {name: safe_open(p, framework="pt") for name, p in bases.items()}
    try:
        with safe_open(vsa, framework="pt") as fv:
            tv_raw = fv.get_tensor("adaln_t_table")
            tables = {name: h.get_tensor("adaln_t_table") for name, h in handles.items()}
            for mod in modules:
                tv, wv = _augment(tv_raw, fv.get_tensor(mod + ".weight"), fv.get_tensor(mod + ".bias"))
                row = {}
                for name, h in handles.items():
                    tb, wb = _augment(tables[name], h.get_tensor(mod + ".weight"),
                                      h.get_tensor(mod + ".bias"))
                    row[name] = _affine_distance(tv, wv, tb, wb)
                out[mod] = row
    finally:
        for h in handles.values():
            h.__exit__(None, None, None)
    return out


def _surface_spectrum(tp, wp) -> list[float]:
    """Singular values of the surface `T' W'.T` without materializing it.

    `A.T A = W' (T'.T T') W'.T` has the same nonzero eigenvalues as
    `G_T^(1/2) G_W G_T^(1/2)`, both 9x9, so the whole spectrum of a
    [1025, 96768] matrix costs two small eigendecompositions.
    """
    gt = tp.T @ tp
    gw = wp.T @ wp
    ev, q = torch.linalg.eigh(gt)
    root = q @ torch.diag(ev.clamp_min(0).sqrt()) @ q.T
    vals = torch.linalg.eigvalsh(root @ gw @ root).clamp_min(0).sqrt()
    return sorted((float(v) for v in vals), reverse=True)


def _principal_angles(a, b) -> list[float]:
    """Cosines of the principal angles between two column spaces, descending."""
    qa, _ = torch.linalg.qr(a)
    qb, _ = torch.linalg.qr(b)
    return sorted((float(v) for v in torch.linalg.svdvals(qa.T @ qb)), reverse=True)


def adaln_structure(vsa: Path, base: Path) -> dict:
    """Where the adaln surfaces differ within their own spectrum.

    The affine distance alone cannot tell "the surface was retrained" from "two
    independent rank-8 fits of the same surface disagree about the directions
    that carry almost no energy". The spectrum and the principal angles can: a
    trained surface moves its dominant components, a refit moves its tail.
    """
    out = {}
    mods = [f"blocks.{i}.adaln_proj.linear" for i in SAMPLE_BLOCKS] + \
           ["final_layer.adaln_proj.linear"]
    with safe_open(vsa, framework="pt") as fv, safe_open(base, framework="pt") as fb:
        tv_raw, tb_raw = fv.get_tensor("adaln_t_table"), fb.get_tensor("adaln_t_table")
        for mod in mods:
            tv, wv = _augment(tv_raw, fv.get_tensor(mod + ".weight"), fv.get_tensor(mod + ".bias"))
            tb, wb = _augment(tb_raw, fb.get_tensor(mod + ".weight"), fb.get_tensor(mod + ".bias"))
            sv, sb = _surface_spectrum(tv, wv), _surface_spectrum(tb, wb)
            out[mod] = {
                "singular_values_vsa": sv,
                "singular_values_base": sb,
                "singular_value_relative_difference":
                    [(x - y) / y if y else None for x, y in zip(sv, sb)],
                "factor_column_space_cosines": _principal_angles(wv, wb),
                "energy_fraction_outside_top3_base":
                    sum(v * v for v in sb[3:]) / sum(v * v for v in sb),
            }
    return out


def gram_identity_control(vsa: Path, base: Path) -> dict:
    """Materialize one adaln surface and check the Gram expansion reproduces it.

    The outcome is genuinely open -- the expansion is a trace identity written
    from scratch, and getting a factor or a transpose wrong would produce a
    plausible-looking number rather than an error. So it earns its cost by this
    repo's rule, where restating a measurement already on screen would not.
    """
    mod = "final_layer.adaln_proj.linear"
    with safe_open(vsa, framework="pt") as fv, safe_open(base, framework="pt") as fb:
        tv, wv = _augment(fv.get_tensor("adaln_t_table"), fv.get_tensor(mod + ".weight"),
                          fv.get_tensor(mod + ".bias"))
        tb, wb = _augment(fb.get_tensor("adaln_t_table"), fb.get_tensor(mod + ".weight"),
                          fb.get_tensor(mod + ".bias"))
    direct_v = tv @ wv.T
    direct_b = tb @ wb.T
    direct = {
        "relative": float((direct_v - direct_b).norm() / direct_b.norm()),
        "cosine": float((direct_v.flatten() @ direct_b.flatten())
                        / (direct_v.norm() * direct_b.norm())),
    }
    via_gram = _affine_distance(tv, wv, tb, wb)
    return {
        "module": mod,
        "materialized_shape": list(direct_v.shape),
        "direct": direct,
        "via_gram": {k: via_gram[k] for k in ("relative", "cosine")},
        "abs_error_relative": abs(direct["relative"] - via_gram["relative"]),
        "abs_error_cosine": abs(direct["cosine"] - via_gram["cosine"]),
    }


# --------------------------------------------------------------------------
# 5. the quantized sample
#
# Compared as `q * s` -- the dequantized weight in the shared rotated basis --
# beside the rounding error that comparison inherits. For round-to-nearest the
# per-element error is uniform on +/- s/2, so E||E||^2 = ncols * sum_r s_r^2 / 12
# from the scale vector alone, with no need to touch the payload. Two
# independent quantizations of the SAME weight would land near
# sqrt(||E_vsa||^2 + ||E_base||^2), and that is the floor each row is read
# against.


def _dequant_rows(handle, key, lo, hi):
    q = handle.get_slice(key)[lo:hi].to(torch.float64)
    s = handle.get_slice(key + "_scale")[lo:hi].to(torch.float64)
    return q * s


def quant_scale_rule(handle, key) -> dict:
    """Whether the file's per-row scale is plain absmax, which fixes whether a
    dequant/requant round trip is a fixed point -- and so whether a byte
    difference in the payload can be read as the weight having moved."""
    q = handle.get_tensor(key)
    rowmax = q.abs().amax(dim=1)
    return {
        "rows": int(rowmax.numel()),
        "fraction_rows_saturating_127": float((rowmax == 127).to(torch.float64).mean()),
        "min_row_absmax_int8": int(rowmax.min()),
    }


def quantized_pair(vh, bh, key) -> dict:
    rows = vh.get_slice(key).get_shape()[0]
    ncols = vh.get_slice(key).get_shape()[1]
    sq_d = sq_b = sq_v = cross = 0.0
    nv_rows, nb_rows = [], []
    for lo in range(0, rows, ROW_CHUNK):
        hi = min(lo + ROW_CHUNK, rows)
        a = _dequant_rows(vh, key, lo, hi)
        b = _dequant_rows(bh, key, lo, hi)
        sq_d += float(((a - b) ** 2).sum())
        sq_b += float((b ** 2).sum())
        sq_v += float((a ** 2).sum())
        cross += float((a * b).sum())
        nv_rows.append(a.norm(dim=1))
        nb_rows.append(b.norm(dim=1))
    nv = torch.cat(nv_rows)
    nb = torch.cat(nb_rows)

    def floor(handle):
        s = handle.get_tensor(key + "_scale").to(torch.float64).flatten()
        return float((ncols * (s ** 2) / 12.0).sum())

    fl_v, fl_b = floor(vh), floor(bh)
    return {
        "shape": [rows, ncols],
        "relative": (sq_d ** 0.5) / (sq_b ** 0.5),
        "cosine": cross / ((sq_v ** 0.5) * (sq_b ** 0.5)),
        "predicted_rounding_floor_relative": ((fl_v + fl_b) ** 0.5) / (sq_b ** 0.5),
        "excess_over_floor": (sq_d ** 0.5) / ((fl_v + fl_b) ** 0.5),
        "row_norm_profile_relative": float((nv - nb).norm() / nb.norm()),
    }


def quantized_sample(vsa: Path, base: Path) -> dict:
    out = {"scale_rule": {}, "per_key": {}}
    with safe_open(vsa, framework="pt") as vh, safe_open(base, framework="pt") as bh:
        for i in SAMPLE_BLOCKS:
            for stem in SAMPLE_LINEARS:
                key = f"blocks.{i}.{stem}.weight"
                out["per_key"][key] = quantized_pair(vh, bh, key)
                out["scale_rule"][key] = {"vsa": quant_scale_rule(vh, key),
                                          "base": quant_scale_rule(bh, key)}
    return out


# --------------------------------------------------------------------------
# 6. the gate itself


def gate_report(vsa: Path) -> dict:
    with safe_open(vsa, framework="pt") as f:
        keys = sorted(k for k in f.keys() if "to_gate_compress" in k)
        blocks = sorted({int(re.search(r"blocks\.(\d+)\.", k).group(1)) for k in keys})
        one = "blocks.0.attn.to_gate_compress"
        desc = bytes(f.get_tensor(one + ".comfy_quant").tolist()).decode()
        shape = f.get_slice(one + ".weight").get_shape()
        # scale vectors are tiny; every block is affordable and depth is the
        # axis worth seeing
        by_block = {}
        for i in blocks:
            s = f.get_tensor(f"blocks.{i}.attn.to_gate_compress.weight_scale").to(torch.float64)
            by_block[i] = {"scale_mean": float(s.mean()), "scale_max": float(s.max())}
        detail = {}
        for i in SAMPLE_BLOCKS:
            gk = f"blocks.{i}.attn.to_gate_compress.weight"
            q = f.get_tensor(gk)
            s = f.get_tensor(gk + "_scale").to(torch.float64)
            w = q.to(torch.float64) * s
            qkv = f.get_tensor(f"blocks.{i}.attn.qkv_proj.weight").to(torch.float64) \
                * f.get_tensor(f"blocks.{i}.attn.qkv_proj.weight_scale").to(torch.float64)
            detail[i] = {
                "frobenius_norm": float(w.norm()),
                "rms_per_element": float(w.pow(2).mean().sqrt()),
                "fraction_int8_exactly_zero": float((q == 0).to(torch.float64).mean()),
                "n_all_zero_rows": int((q.abs().amax(dim=1) == 0).sum()),
                "rms_ratio_to_qkv_proj": float(w.pow(2).mean().sqrt() / qkv.pow(2).mean().sqrt()),
                # The gate multiplies the coarse branch's output directly, so the
                # magnitude of `W_gate @ x` IS the mixing coefficient. The row
                # norm is the half of that product this file can supply; the
                # activation half is not in any checkpoint.
                "mean_row_norm": float(w.norm(dim=1).mean()),
                "mean_row_norm_qkv_proj": float(qkv.norm(dim=1).mean()),
                "saturation": quant_scale_rule(f, gk),
            }
        return {
            "key_patterns": dict(sorted(Counter(collapse(k) for k in keys).items())),
            "n_gate_modules": len({k.rsplit(".", 1)[0] for k in keys}),
            "blocks_carrying_a_gate": {"min": blocks[0], "max": blocks[-1], "count": len(blocks)},
            "token_refiner_blocks_carrying_a_gate":
                sorted(k for k in keys if k.startswith("token_refiner.")),
            "weight_shape": shape,
            "has_bias": any(k.endswith("to_gate_compress.bias") for k in keys),
            "comfy_quant_descriptor": desc,
            "quant_descriptor_matches_qkv_proj":
                desc == bytes(f.get_tensor("blocks.0.attn.qkv_proj.comfy_quant").tolist()).decode(),
            "scale_by_block": by_block,
            "sampled_blocks": detail,
        }


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Inventory and divergence of the FastVideo VSA H3 checkpoint.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    needed = [VSA, FL2VA_PRUNED, REF2VA_PRUNED, FL2VA_UNPRUNED]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        print("ABSENT, cannot run:", missing)
        return 1

    bases = {"fl2va_pruned": FL2VA_PRUNED, "ref2va_pruned": REF2VA_PRUNED}

    record = {
        "question": "what is in the FastVideo VSA-distilled H3 checkpoint, which "
                    "partition it is built on, and how far its weights moved",
        "artifact": "models/diffusion_models/" + VSA.name,
        "compared_against": {k: "models/diffusion_models/" + v.name for k, v in bases.items()},
        "inventory": {"vsa": inventory(VSA), "fl2va_pruned": inventory(FL2VA_PRUNED)},
        "keyset_diff_vs_fl2va_pruned": keyset_diff(VSA, FL2VA_PRUNED),
        "keyset_diff_vs_fl2va_unpruned": keyset_diff(VSA, FL2VA_UNPRUNED),
    }

    print("== inventory ==")
    inv = record["inventory"]["vsa"]
    print(f"  {inv['tensors']} tensors, {inv['parameters']:,} parameters, "
          f"{inv['bytes_on_disk'] / 2**30:.2f} GiB, metadata: {inv['safetensors_metadata']}")
    print(f"  gate modules: {len(inv['gate_modules'])}")

    print("== key set vs pruned fl2va ==")
    kd = record["keyset_diff_vs_fl2va_pruned"]
    print(f"  only in VSA: {kd['only_in_left']}")
    print(f"  only in base: {kd['only_in_right']}")
    print(f"  shape/dtype changed: {kd['shape_or_dtype_changed']}")

    print("== byte census ==")
    record["byte_census"] = {
        "vsa_vs_fl2va_pruned": byte_census(VSA, FL2VA_PRUNED),
        "vsa_vs_ref2va_pruned": byte_census(VSA, REF2VA_PRUNED),
        "control_fl2va_pruned_vs_unpruned": byte_census(FL2VA_PRUNED, FL2VA_UNPRUNED),
    }
    for name, c in record["byte_census"].items():
        print(f"  {name:36s} match {c['n_matching_prefix']:4d}  differ {c['n_differing']:4d}"
              f"  incomparable {c['n_incomparable_shape_or_dtype']:4d}")

    print("== unquantized fingerprint ==")
    fp = fingerprint(VSA, bases)
    norms = fp.pop("_block_norms")
    record["unquantized_fingerprint"] = fp
    record["block_norm_summary"] = summarise_norms(norms, bases)
    for key, row in fp.items():
        if "absent_from_vsa" in row:
            print(f"  {key:48s} ABSENT")
            continue
        a, b = row["fl2va_pruned"], row["ref2va_pruned"]
        print(f"  {key:48s} fl2va {a['relative']:.3e}  ref2va {b['relative']:.3e}"
              f"   {'identical' if a['identical'] else ''}")
    for name, s in record["block_norm_summary"].items():
        print(f"  block norms vs {name:16s} n={s['n']} bit-identical={s['n_bit_identical']} "
              f"median rel {s['relative']['median']:.3e}")

    print("== representation floor control ==")
    record["representation_floor_bf16_roundtrip"] = representation_floor(FL2VA_PRUNED)
    for k, v in record["representation_floor_bf16_roundtrip"].items():
        print(f"  {k:48s} bf16 round trip costs {v:.3e}")

    print("== adaln affine map (basis-invariant) ==")
    record["gram_identity_control"] = gram_identity_control(VSA, FL2VA_PRUNED)
    g = record["gram_identity_control"]
    print(f"  control: direct rel {g['direct']['relative']:.6e} vs gram "
          f"{g['via_gram']['relative']:.6e}   abs err {g['abs_error_relative']:.3e}")
    aff = adaln_affine(VSA, bases)
    record["adaln_affine"] = aff
    for name in bases:
        rels = sorted(v[name]["relative"] for v in aff.values())
        amps = sorted(v[name]["factor_to_product_amplification"] for v in aff.values())
        print(f"  vs {name:16s} n={len(rels)} min {rels[0]:.3e} "
              f"median {rels[len(rels) // 2]:.3e} max {rels[-1]:.3e}"
              f"   amplification median {amps[len(amps) // 2]:.1f}")

    record["adaln_structure"] = adaln_structure(VSA, FL2VA_PRUNED)
    for mod, d in record["adaln_structure"].items():
        sv, sb = d["singular_values_vsa"], d["singular_values_base"]
        top = ", ".join(f"{x:.3e}" for x in sb[:4])
        rel = ", ".join(f"{x:+.1e}" for x in d["singular_value_relative_difference"][:4])
        print(f"  {mod}")
        print(f"     base top-4 sv: {top}   tail energy beyond top-3: "
              f"{d['energy_fraction_outside_top3_base']:.2e}")
        print(f"     vsa/base sv relative diff, top-4: {rel}")
        print(f"     factor column-space cosines: "
              f"{', '.join(f'{c:.3f}' for c in d['factor_column_space_cosines'])}")

    print("== quantized sample vs pruned fl2va ==")
    qs = quantized_sample(VSA, FL2VA_PRUNED)
    record["quantized_sample"] = qs
    vsat = [v["vsa"]["fraction_rows_saturating_127"] for v in qs["scale_rule"].values()]
    bsat = [v["base"]["fraction_rows_saturating_127"] for v in qs["scale_rule"].values()]
    print(f"  rows saturating at |q|=127, over {len(vsat)} sampled linears: "
          f"vsa {min(vsat):.4f}-{max(vsat):.4f}, base {min(bsat):.4f}-{max(bsat):.4f}")
    for key, v in qs["per_key"].items():
        print(f"  {key:40s} rel {v['relative']:.4e}  floor "
              f"{v['predicted_rounding_floor_relative']:.4e}  "
              f"excess {v['excess_over_floor']:.3f}  rownorm {v['row_norm_profile_relative']:.4e}")

    print("== gate ==")
    gr = gate_report(VSA)
    record["gate"] = gr
    print(f"  {gr['n_gate_modules']} modules, shape {gr['weight_shape']}, bias {gr['has_bias']}")
    print(f"  descriptor {gr['comfy_quant_descriptor']}")
    print(f"  same descriptor as qkv_proj: {gr['quant_descriptor_matches_qkv_proj']}")
    print(f"  token_refiner gates: {gr['token_refiner_blocks_carrying_a_gate']}")
    for i, d in gr["sampled_blocks"].items():
        print(f"  block {i:2d}: rms {d['rms_per_element']:.4e}  "
              f"rms/qkv {d['rms_ratio_to_qkv_proj']:.2e}  "
              f"row norm {d['mean_row_norm']:.4e} (qkv {d['mean_row_norm_qkv_proj']:.4e})  "
              f"zero int8 {d['fraction_int8_exactly_zero']:.4f}  "
              f"all-zero rows {d['n_all_zero_rows']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
