#!/usr/bin/env python3
"""Bake a PDD backbone into a new int8_convrot checkpoint, offline.

The plan and the reasons are `docs/research/pdd/2026-09-05_bake_plan.md`;
this file is its script and restates none of it. Three modes, run in this
order, each a precondition of the next:

  --control     read-only identity control. Quantise each module's RELEASE
                weight with NO delta through the exact pipeline this script
                bakes with (hf_to_comfy, the rotation, `gs` off `comfy_quant`,
                round-to-nearest) and compare int8 codes AND fp32 row scales
                bit for bit against the shipped file. Reports per kind. Codes
                matching proves the pipeline independent of the LoRA; codes
                not matching names the vendor's regime before anything is
                written. Nothing is written except the result record.
  --scratch-block N --out X
                bake ONE block's four modules at --strength to a small
                safetensors, reopen it from disk, dequantise, and compare
                against W_release + d. First write, first test.
  --out X       the full bake: every non-backbone key copied byte for byte
                from --base, every int8 backbone module rebuilt as
                Q_rtn(hf_to_comfy(W_release) + strength * delta), metadata
                stamped, written to X.partial and renamed only after the
                reopened file passes the population and error checks.

One authority per operation, imported not restated: the release-to-ComfyUI
mapping is `analyze_quant_delta.hf_to_comfy`; the quantiser is comfy-kitchen's
`quantize_int8_convrot_weight`, the same call `measure_bake_realisation.py`
made on 2026-08-31; the backbone delta is parsed by the node's own
`pdd_lora.split_unmerged` and scaled by the convention `comfy.lora` applies
(strength * alpha / rank); the population is `vendor_config.transformer_depth()`
times `pdd_lora.BACKBONE_KINDS`, never counted off an input.

Memory: one module resident at a time. The writer precomputes the header
from --base's own shapes and dtypes (unchanged by the bake) and streams the
data section in header order, copying raw bytes for everything it does not
rebuild. The script reads `available` from /proc/meminfo and refuses to start
below --rss-budget-gb, because the live server holds most of this host while
a DiT is staged and the 2026-08-25 bridge load took it down.

CPU only. Run with CUDA_VISIBLE_DEVICES= (docs/checks.md, "Running them").

    CUDA_VISIBLE_DEVICES= python bench/bake_pdd_checkpoint.py \\
        --base models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \\
        --reference <release>/FL2VA/transformer \\
        --lora models/loras/h3/minimax_h3_fl2va_pdd_8step_comfy.safetensors \\
        --control --result bench/results/<date>_bake_identity_control.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import resource
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))             # this repo
sys.path.insert(0, str(_HERE.parents[2]))         # ComfyUI root

from analyze_checkpoint_delta import header  # noqa: E402
from analyze_quant_delta import Reference, hf_to_comfy, head_dim, marker  # noqa: E402
import pdd_lora as P  # noqa: E402
import vendor_config  # noqa: E402
from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    dequantize_int8_convrot_weight, quantize_int8_convrot_weight)

KINDS = P.BACKBONE_KINDS
CHUNK = 64 << 20
#: Host-memory floor to start under, GB. REASONED: about twice the peak RSS
#: the first runs recorded (`bench/results/2026-09-05_bake_*.json`,
#: `peak_rss_gb`), so a run that has to share the host with a staged DiT is
#: refused before it starts rather than killed mid-write.
RSS_BUDGET_GB = 12.0
#: The rename gate's error bound on the reopened file, relative Frobenius
#: against `W_release + d` per module. REASONED: the shipped checkpoint's own
#: per-module error and every baked module's sit near one hundredth
#: (`2026-09-05_bake_fl2va_full.json`, `reopened.sampled`); a wrong group
#: size, a wrong row order or a scale written in the wrong dtype land an
#: order of magnitude above this, and a regime tie does not move it. Until
#: 2026-09-05 evening the gate was population and per-kind count only, which
#: the interrupted code review named: a plausibly wrong bake renamed clean.
ERR_VS_TARGET_MAX = 0.05


# ----------------------------------------------------------------- helpers

def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1 << 20)
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(CHUNK), b""):
            h.update(buf)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_HERE.parent,
            text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def raw_bytes(path: str, base: int, info: dict) -> bytes:
    o0, o1 = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + o0)
        return f.read(o1 - o0)


def stored_codes(path: str, hdr: dict, base: int, mod: str):
    """(int8 codes, fp32 row scales) exactly as the file stores them."""
    q = np.frombuffer(raw_bytes(path, base, hdr[mod + ".weight"]),
                      dtype=np.int8).reshape(hdr[mod + ".weight"]["shape"])
    s = np.frombuffer(raw_bytes(path, base, hdr[mod + ".weight_scale"]),
                      dtype=np.float32).reshape(hdr[mod + ".weight_scale"]["shape"])
    return q, s


def quantise(w: np.ndarray, gs: int):
    """Round-to-nearest through the shipped quantiser. Returns (codes, scales)
    as numpy, dtypes matching the shipped file's I8 and F32."""
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    q, s = quantize_int8_convrot_weight(t, gs, stochastic_rounding=None)
    return q.numpy(), s.to(torch.float32).numpy()


def dequantise(q: np.ndarray, s: np.ndarray, gs: int) -> np.ndarray:
    return dequantize_int8_convrot_weight(
        torch.from_numpy(q.astype(np.int8)), torch.from_numpy(s.astype(np.float32)),
        gs).numpy()


def rel(a: np.ndarray, b: np.ndarray) -> float:
    d = (a - b).astype(np.float64)
    return float(np.sqrt((d * d).sum()) / max(np.sqrt((a.astype(np.float64) ** 2).sum()), 1e-12))


def expected_population() -> tuple[int, int, list[str]]:
    depth, refiner = vendor_config.transformer_depth()
    mods = [f"blocks.{i}.{k}" for i in range(depth) for k in KINDS]
    return depth, refiner, mods


def load_deltas(lora: Path, depth: int) -> dict[str, tuple]:
    """Every backbone module's (A, B, alpha, rank), parsed by the node's own
    `split_unmerged` so the key grammar cannot drift from what the node
    applies. The scale strength * alpha / rank is `comfy.lora`'s convention
    for a LoRA patch and what the node folds into `b` at the call."""
    raw = {}
    with safe_open(str(lora), "pt") as f:
        for k in f.keys():
            if k.startswith("diffusion_model.blocks."):
                raw[k] = f.get_tensor(k)
    _, lifted = P.split_unmerged(raw, range(depth), kinds=KINDS)
    out = {}
    for (i, kind), (a, b, alpha, rank) in lifted.items():
        out[f"blocks.{i}.{kind}"] = (a, b, alpha, rank)
    if len(out) != depth * len(KINDS):
        raise SystemExit(f"{lora.name}: {len(out)} backbone modules parsed "
                         f"against {depth * len(KINDS)} the release declares.")
    return out


def delta_of(entry, strength: float) -> np.ndarray:
    a, b, alpha, rank = entry
    scale = strength * alpha / rank
    return (scale * (b.to(torch.float32) @ a.to(torch.float32))).numpy()


def sample_modules(mods: list[str], every: int) -> list[str]:
    return mods[::every] if every > 1 else list(mods)


#: The identity control's pass criterion. Provenance: MEASURED, TIES. On
#: 2026-09-05 the strict bit-identity failed on every module of block 0 by a
#: handful of codes out of tens of millions, each off by exactly one step,
#: with scales differing at float32-ulp level and the error against the
#: release equal to the shipped file's within float epsilon
#: (`bench/results/2026-09-05_bake_identity_control_block0.json`, then all
#: 200 in `..._bake_identity_control.json`): round-to-nearest on another
#: device or accumulation order, with a few elements landing on rounding
#: ties. The alternatives were run and ruled out the same day (stochastic
#: rounding and every bf16 path differ on millions of codes). So "same
#: regime" is: every code within one step, a differing fraction and a scale
#: deviation both below these bounds, and the two errors equal within float
#: epsilon. A strict bit-identity verdict would read as a regime mismatch to
#: the next reader, which is the wrong reading. Cross-backend: the 2026-08-31
#: regimes record has eager and CUDA round-to-nearest agreeing exactly, so
#: the ties are between the vendor's run and ours, not between our backends.
TIES_CRITERION = {
    "codes_max_abs_diff_le": 1,
    "codes_differing_frac_lt": 1e-6,
    "scale_rel_max_lt": 1e-6,
    "err_control_vs_shipped_abs_lt": 1e-6,
    "provenance": "measured, ties (2026-09-05, block 0, alternatives ruled out)",
}


def ties_verdict(rows: list[dict]) -> dict:
    """Grade identity-control rows against TIES_CRITERION."""
    c = TIES_CRITERION
    failing = [r["module"] for r in rows if not (
        r["codes_max_abs_diff"] <= c["codes_max_abs_diff_le"]
        and r["codes_differing_frac"] < c["codes_differing_frac_lt"]
        and r.get("scale_rel_row_max", r["scale_rel_max"]) < c["scale_rel_max_lt"]
        and abs(r["err_control"] - r["err_shipped"]) < c["err_control_vs_shipped_abs_lt"])]
    return {"criterion": c, "same_regime": not failing, "failing_modules": failing,
            "bit_identical_modules": sum(r["codes_equal"] and r["scales_equal"] for r in rows),
            "modules": len(rows),
            "codes_differing_frac_max": max(r["codes_differing_frac"] for r in rows),
            "codes_max_abs_diff": max(r["codes_max_abs_diff"] for r in rows),
            "scale_rel_max": max(r["scale_rel_max"] for r in rows),
            "scale_rel_row_max": max((r.get("scale_rel_row_max", r["scale_rel_max"]) for r in rows)),
            "err_gap_max": max(abs(r["err_control"] - r["err_shipped"]) for r in rows),
            "reading": ("round-to-nearest, same regime as the shipped file; the "
                        "differing codes are rounding ties" if not failing else
                        "outside the ties criterion; do not bake before this is understood")}


# ------------------------------------------------------------- the control

def run_control(args, hdr, base, ref, mods, gs_of, hd) -> dict:
    """Strength-zero identity: does Q_rtn(hf_to_comfy(W_release)) reproduce
    the shipped codes and scales bit for bit, per module?"""
    rows = []
    t0 = time.perf_counter()
    for mod in mods:
        gs = gs_of[mod]
        w_ref = hf_to_comfy(mod + ".weight", ref.get(mod + ".weight"), hd)
        q_new, s_new = quantise(w_ref, gs)
        q_old, s_old = stored_codes(args.base, hdr, base, mod)
        if q_new.shape != q_old.shape or s_new.shape != s_old.shape:
            raise SystemExit(f"{mod}: shape mismatch, new {q_new.shape}/{s_new.shape} "
                             f"against stored {q_old.shape}/{s_old.shape}")
        codes_equal = bool(np.array_equal(q_new, q_old))
        scales_equal = bool(np.array_equal(s_new.view(np.int32), s_old.view(np.int32)))
        diff = (q_new.astype(np.int16) - q_old.astype(np.int16))
        n_diff = int((diff != 0).sum())
        row = {
            "module": mod, "kind": ".".join(mod.split(".")[2:]), "groupsize": gs,
            "codes_equal": codes_equal, "scales_equal": scales_equal,
            "codes_differing": n_diff,
            "codes_differing_frac": n_diff / diff.size,
            "codes_max_abs_diff": int(np.abs(diff).max()) if n_diff else 0,
            "scale_rel_max": float(np.abs(s_new - s_old).max() / max(np.abs(s_old).max(), 1e-30)),
            # per row, so a small-scale row off by a large fraction of itself
            # cannot hide under the global maximum (interrupted review)
            "scale_rel_row_max": float((np.abs(s_new - s_old) / np.maximum(np.abs(s_old), 1e-30)).max()),
            # the error each lands at against the release, for the reader who
            # wants the magnitude beside the identity verdict
            "err_shipped": rel(w_ref, dequantise(q_old, s_old, gs)),
            "err_control": rel(w_ref, dequantise(q_new, s_new, gs)),
        }
        rows.append(row)
        print(f"  {mod:28s} codes {'EQUAL' if codes_equal else f'differ {n_diff} ({row['codes_differing_frac']:.4f}) max|d|={row['codes_max_abs_diff']}':40s} "
              f"scales {'EQUAL' if scales_equal else f'differ rel {row['scale_rel_max']:.2e}'}", flush=True)
        del w_ref, q_new, s_new, q_old, s_old, diff
    per_kind = {}
    for k in KINDS:
        ks = [r for r in rows if r["kind"] == k]
        per_kind[k] = {"modules": len(ks),
                       "codes_equal": sum(r["codes_equal"] for r in ks),
                       "scales_equal": sum(r["scales_equal"] for r in ks),
                       "codes_differing_frac_max": max((r["codes_differing_frac"] for r in ks), default=None),
                       "codes_max_abs_diff": max((r["codes_max_abs_diff"] for r in ks), default=None)}
    return {"rows": rows, "per_kind": per_kind,
            "all_codes_equal": all(r["codes_equal"] for r in rows),
            "all_scales_equal": all(r["scales_equal"] for r in rows),
            "wall_s": time.perf_counter() - t0}


# --------------------------------------------------------------- the bake

def plan_header(hdr: dict, meta: dict) -> tuple[bytes, dict]:
    """The output header: --base's names, dtypes and shapes verbatim, offsets
    recomputed in name order, our metadata in front."""
    out = {"__metadata__": meta}
    off = 0
    order = sorted(hdr)
    for name in order:
        info = hdr[name]
        n = info["data_offsets"][1] - info["data_offsets"][0]
        out[name] = {"dtype": info["dtype"], "shape": info["shape"],
                     "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(out, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    return hb, out


def bake_module(mod, ref, hd, gs, deltas, strength):
    w_ref = hf_to_comfy(mod + ".weight", ref.get(mod + ".weight"), hd)
    target = w_ref + delta_of(deltas[mod], strength) if strength != 0.0 else w_ref
    q, s = quantise(target, gs)
    return q, s, target


def write_bake(args, hdr, base, ref, mods, gs_of, hd, deltas, meta,
               only_mods=None) -> dict:
    """Stream the output. `only_mods` restricts the output to those modules'
    tensors (the scratch-block mode); otherwise every key of --base."""
    baked = set(mods if only_mods is None else only_mods)
    names_out = (sorted(hdr) if only_mods is None
                 else sorted(n for n in hdr
                             if any(n.startswith(m + ".") for m in baked)))
    sub_hdr = {n: hdr[n] for n in names_out}
    hb, out_hdr = plan_header(sub_hdr, meta)
    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".partial")
    written_by_kind = {k: 0 for k in KINDS}
    errs = []
    t0 = time.perf_counter()
    with open(tmp, "wb") as w:
        w.write(struct.pack("<Q", len(hb)))
        w.write(hb)
        pending = {}                      # computed tensors waiting their slot
        for name in names_out:
            mod: str | None = None
            for m in baked:
                if name == m + ".weight" or name == m + ".weight_scale":
                    mod = m
                    break
            if mod is None:
                info = hdr[name]
                o0, o1 = info["data_offsets"]
                with open(args.base, "rb") as f:
                    f.seek(base + o0)
                    left = o1 - o0
                    while left:
                        buf = f.read(min(CHUNK, left))
                        w.write(buf)
                        left -= len(buf)
                continue
            if mod not in pending:
                q, s, target = bake_module(mod, ref, hd, gs_of[mod], deltas, args.strength)
                errs.append({"module": mod,
                             "err_vs_target": rel(target, dequantise(q, s, gs_of[mod]))})
                pending[mod] = {"weight": q, "weight_scale": s}
                del target
                kind = ".".join(mod.split(".")[2:])
                written_by_kind[kind] += 1
                print(f"  baked {mod:28s} err_vs_target {errs[-1]['err_vs_target']:.5f}", flush=True)
            assert mod is not None
            slot = name[len(mod) + 1:]
            arr = pending[mod].pop(slot)
            expect = out_hdr[name]["data_offsets"][1] - out_hdr[name]["data_offsets"][0]
            buf = np.ascontiguousarray(arr).tobytes()
            if len(buf) != expect:
                raise SystemExit(f"{name}: {len(buf)} bytes computed against {expect} in the header")
            w.write(buf)
            if not pending[mod]:
                del pending[mod]
    wall = time.perf_counter() - t0
    return {"tmp": tmp, "out": out, "written_by_kind": written_by_kind,
            "errs": errs, "wall_s": wall, "names": len(names_out)}


def verify_reopened(path: Path, hdr_expect: dict, mods, gs_of, ref, hd,
                    deltas, strength, depth, every: int) -> dict:
    """The artifact on disk is the thing under test."""
    hdr2, base2 = header(str(path))
    if set(hdr2) != set(hdr_expect):
        missing = sorted(set(hdr_expect) - set(hdr2))[:5]
        extra = sorted(set(hdr2) - set(hdr_expect))[:5]
        raise SystemExit(f"{path.name}: key set differs; missing {missing}, extra {extra}")
    for n, info in hdr_expect.items():
        if hdr2[n]["dtype"] != info["dtype"] or hdr2[n]["shape"] != info["shape"]:
            raise SystemExit(f"{path.name}: {n} is {hdr2[n]['dtype']} {hdr2[n]['shape']} "
                             f"against {info['dtype']} {info['shape']}")
    i8 = [n for n, i in hdr2.items() if i["dtype"] == "I8" and n.endswith(".weight")]
    by_kind = {k: sum(1 for n in i8 if n.endswith("." + k + ".weight")) for k in KINDS}
    want = {k: depth for k in KINDS}
    pop_ok = by_kind == want and len(i8) == depth * len(KINDS)
    checks = []
    for mod in sample_modules(mods, every):
        q, s = stored_codes(str(path), hdr2, base2, mod)
        gs = gs_of[mod]
        w_ref = hf_to_comfy(mod + ".weight", ref.get(mod + ".weight"), hd)
        target = w_ref + delta_of(deltas[mod], strength) if strength != 0.0 else w_ref
        checks.append({"module": mod,
                       "err_vs_target": rel(target, dequantise(q, s, gs)),
                       "err_vs_release": rel(w_ref, dequantise(q, s, gs))})
        del q, s, w_ref, target
    return {"population_by_kind": by_kind, "population_ok": pop_ok,
            "int8_modules": len(i8), "sampled": checks,
            "err_vs_target_max": max(c["err_vs_target"] for c in checks) if checks else None}


# ------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="the shipped pruned int8_convrot checkpoint, the template")
    ap.add_argument("--reference", required=True, type=Path, help="<release>/FL2VA/transformer")
    ap.add_argument("--lora", required=True, type=Path, help="the converted FULL PDD sidecar")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--control", action="store_true", help="read-only identity control at strength 0")
    ap.add_argument("--regrade", type=Path, default=None,
                    help="grade an existing identity-control record against TIES_CRITERION in place; nothing else runs")
    ap.add_argument("--scratch-block", type=int, default=None, help="bake one block's modules to --out")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--result", type=Path, default=None, help="JSON record (required except with --regrade)")
    ap.add_argument("--blocks", default="all", help="control only: block subset, e.g. 0,49")
    ap.add_argument("--verify-every", type=int, default=1,
                    help="full bake: dequantise-and-compare every Nth module on reopen (1 = all)")
    ap.add_argument("--rss-budget-gb", type=float, default=RSS_BUDGET_GB,
                    help="refuse to start when MemAvailable is below this")
    ap.add_argument("--hash-shards", action="store_true",
                    help="full bake: sha256 every release shard into the metadata (slow, read-only)")
    args = ap.parse_args()

    if args.regrade is not None:
        rec = json.loads(args.regrade.read_text())
        if rec.get("mode") != "identity_control":
            raise SystemExit(f"{args.regrade} is not an identity-control record")
        rec["verdict"] = ties_verdict(rec["modules"])
        rec["regraded_by_commit"] = git_commit()
        args.regrade.write_text(json.dumps(rec, indent=2) + "\n")
        print(f"{args.regrade}: {rec['verdict']['reading']}; "
              f"bit-identical {rec['verdict']['bit_identical_modules']}/{rec['verdict']['modules']}, "
              f"max|d| {rec['verdict']['codes_max_abs_diff']}, "
              f"differing frac max {rec['verdict']['codes_differing_frac_max']:.2e}, "
              f"scale rel max {rec['verdict']['scale_rel_max']:.2e}, err gap max {rec['verdict']['err_gap_max']:.2e}")
        return 0 if rec["verdict"]["same_regime"] else 3
    if torch.cuda.is_available():
        raise SystemExit("a CUDA device is visible; run with CUDA_VISIBLE_DEVICES= "
                         "(docs/checks.md, 'Running them')")
    avail = mem_available_gb()
    if avail < args.rss_budget_gb:
        raise SystemExit(f"MemAvailable {avail:.1f} GB is below the budget "
                         f"{args.rss_budget_gb:.1f} GB; the live server may be "
                         f"holding the host. Refusing to start.")
    if args.result is None:
        raise SystemExit("--result is required")
    modes = sum(bool(x) for x in (args.control, args.scratch_block is not None,
                                  args.out is not None and args.scratch_block is None))
    if modes != 1:
        raise SystemExit("pick exactly one of --control, --scratch-block N --out X, --out X")
    if (args.scratch_block is not None or not args.control) and args.out is None:
        raise SystemExit("--out is required to write anything")

    hdr, base = header(args.base)
    ref = Reference(args.reference)
    hd = head_dim(hdr)
    depth, refiner, mods = expected_population()
    # The three inputs must belong together, and nothing else refuses a
    # mismatch: the sidecar names the pruned base it was solved against and
    # its partition, and the release folder names its partition in its path.
    # A ref2va sidecar on the fl2va base would otherwise bake and verify
    # against the same wrong target (interrupted review, 2026-09-05).
    with safe_open(str(args.lora), "pt") as f:
        lora_meta = f.metadata() or {}
    want_base = lora_meta.get("h3_pdd_pruned_base") or lora_meta.get("h3_pdd_base")
    if want_base and want_base != Path(args.base).name:
        raise SystemExit(f"{args.lora.name} was converted against {want_base}; "
                         f"--base is {Path(args.base).name}. Refusing.")
    part = ("fl2va" if "fl2va" in Path(args.base).name
            else "ref2va" if "ref2va" in Path(args.base).name else None)
    if part is None:
        raise SystemExit(f"cannot read a partition off {Path(args.base).name}")
    ref_parts = {x.lower() for x in args.reference.resolve().parts}
    if part not in ref_parts:
        raise SystemExit(f"--reference {args.reference} does not name the "
                         f"{part} partition that --base is; refusing to bake "
                         f"one partition's weights with the other's release.")
    src = lora_meta.get("h3_pdd_source", "")
    if src and part.upper() not in src.upper():
        raise SystemExit(f"{args.lora.name} was converted from {src}, not a "
                         f"{part} source; refusing.")
    missing = [m for m in mods if m + ".weight" not in hdr or hdr[m + ".weight"]["dtype"] != "I8"]
    if missing:
        raise SystemExit(f"{Path(args.base).name} lacks int8 weights for {len(missing)} "
                         f"declared modules, e.g. {missing[0]}")
    gs_of = {}
    for m in mods:
        conf = marker(args.base, hdr, base, m) or {}
        if not conf.get("convrot") or "convrot_groupsize" not in conf:
            raise SystemExit(f"{m}: comfy_quant declares no convrot groupsize: {conf}")
        gs_of[m] = int(conf["convrot_groupsize"])
    if args.blocks != "all":
        want = {int(x) for x in args.blocks.split(",")}
        mods_run = [m for m in mods if int(m.split(".")[1]) in want]
    else:
        mods_run = mods

    record = {
        "measured": _dt.date.today().isoformat(),
        "produced_by": "bench/bake_pdd_checkpoint.py",
        "commit": git_commit(),
        "base": Path(args.base).name,
        "reference": "/".join(args.reference.parts[-3:]),
        "lora": args.lora.name,
        "lora_sha256": None,
        "strength": args.strength,
        "rounding": "round_to_nearest",
        "population": {"depth": depth, "refiner": refiner, "kinds": list(KINDS),
                       "modules": len(mods)},
        "groupsizes": sorted(set(gs_of.values())),
        "mem_available_gb_at_start": avail,
        "is_not": ("a runtime or perceptual measurement. Stored weights only; "
                   "int8_convrot is W8A8 and the activation rounding is untouched."),
    }

    if args.control:
        print(f"identity control: {len(mods_run)} modules at strength 0, read-only", flush=True)
        res = run_control(args, hdr, base, ref, mods_run, gs_of, hd)
        record.update({"mode": "identity_control", "modules_run": len(mods_run),
                       "all_codes_equal": res["all_codes_equal"],
                       "all_scales_equal": res["all_scales_equal"],
                       "per_kind": res["per_kind"], "modules": res["rows"],
                       "wall_s": res["wall_s"], "peak_rss_gb": peak_rss_gb()})
        record["verdict"] = ties_verdict(res["rows"])
        args.result.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\n{len(mods_run)} modules -> {args.result}")
        for k, v in res["per_kind"].items():
            print(f"  {k:14s} codes equal {v['codes_equal']}/{v['modules']}  "
                  f"scales equal {v['scales_equal']}/{v['modules']}  "
                  f"max differing frac {v['codes_differing_frac_max']}  max|d| {v['codes_max_abs_diff']}")
        print(f"  verdict: {record['verdict']['reading']}")
        print(f"  wall {res['wall_s']:.1f}s  peak RSS {record['peak_rss_gb']:.2f} GB")
        return 0 if record["verdict"]["same_regime"] else 3

    record["lora_sha256"] = sha256_file(args.lora)
    deltas = load_deltas(args.lora, depth)
    meta = {
        "h3_bake_produced_by": "bench/bake_pdd_checkpoint.py",
        "h3_bake_commit": record["commit"],
        "h3_bake_date": record["measured"],
        "h3_bake_base": record["base"],
        "h3_bake_reference": record["reference"],
        "h3_bake_lora": record["lora"],
        "h3_bake_lora_sha256": record["lora_sha256"],
        "h3_bake_strength": repr(args.strength),
        "h3_bake_rounding": record["rounding"],
        "h3_bake_groupsizes": ",".join(str(g) for g in record["groupsizes"]),
        "h3_bake_partition": part,
        "h3_bake_modules": str(len(mods)),
    }
    if args.hash_shards:
        shards = sorted(set(ref.map.values()))
        meta["h3_bake_reference_shards_sha256"] = json.dumps(
            {s: sha256_file(args.reference / s) for s in shards})

    if args.scratch_block is not None:
        only = [m for m in mods if int(m.split(".")[1]) == args.scratch_block]
        print(f"scratch bake: block {args.scratch_block}, {len(only)} modules at strength {args.strength}", flush=True)
        meta["h3_bake_scratch_block"] = str(args.scratch_block)
        w = write_bake(args, hdr, base, ref, mods, gs_of, hd, deltas, meta, only_mods=only)
        w["tmp"].rename(w["out"])
        sub_hdr = {n: hdr[n] for n in hdr if any(n.startswith(m + ".") for m in only)}
        v = verify_reopened(w["out"], sub_hdr, only, gs_of, ref, hd, deltas,
                            args.strength, depth, every=1)
        v["population_ok"] = v["int8_modules"] == len(only)     # a block, not the DiT
        record.update({"mode": "scratch_block", "block": args.scratch_block,
                       "out": str(w["out"]), "written_by_kind": w["written_by_kind"],
                       "bake_errs": w["errs"], "reopened": v,
                       "wall_s": w["wall_s"], "peak_rss_gb": peak_rss_gb()})
        args.result.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nreopened {w['out']}: int8 modules {v['int8_modules']}, "
              f"max err vs target {v['err_vs_target_max']:.5f}, wall {w['wall_s']:.1f}s, "
              f"peak RSS {record['peak_rss_gb']:.2f} GB -> {args.result}")
        return 0 if v["population_ok"] else 3

    print(f"full bake: {len(mods)} modules at strength {args.strength} -> {args.out}", flush=True)
    w = write_bake(args, hdr, base, ref, mods, gs_of, hd, deltas, meta)
    v = verify_reopened(w["tmp"], hdr, mods, gs_of, ref, hd, deltas,
                        args.strength, depth, every=args.verify_every)
    record.update({"mode": "full_bake", "out": str(w["out"]),
                   "written_by_kind": w["written_by_kind"], "bake_errs": w["errs"],
                   "reopened": v, "metadata": meta,
                   "wall_s": w["wall_s"], "peak_rss_gb": peak_rss_gb()})
    err_ok = (v["err_vs_target_max"] is not None
              and v["err_vs_target_max"] < ERR_VS_TARGET_MAX)
    ok = (v["population_ok"] and w["written_by_kind"] == {k: depth for k in KINDS}
          and err_ok)
    record["err_gate"] = {"bound": ERR_VS_TARGET_MAX, "max": v["err_vs_target_max"],
                          "ok": err_ok}
    if ok:
        w["tmp"].rename(w["out"])
        record["out_sha256"] = sha256_file(w["out"])
    else:
        record["refused"] = ("population, per-kind count or error gate failed; "
                             ".partial left in place")
    args.result.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\n{'WROTE' if ok else 'REFUSED'} {w['out']}: population {v['population_by_kind']}, "
          f"max err vs target {v['err_vs_target_max']}, wall {w['wall_s']:.1f}s, "
          f"peak RSS {record['peak_rss_gb']:.2f} GB -> {args.result}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
