#!/usr/bin/env python3
"""What separates the two shipped quantizations of the same H3 DiT: the
`fp8_scaled` file against the `int8_convrot` file, and -- when the bf16
release is on disk -- each of them against the weights they were made from.

## The two formats, read from the files rather than assumed

Both pruned checkpoints quantize the same 200 block linears (`attn.qkv_proj`,
`attn.out_proj`, `mlp.fc1`, `mlp.fc2` over 50 blocks) and leave the same 274
weight tensors alone (`token_refiner` entirely, the patch and output
projections, the norms, and the curve-form AdaLN). Each quantized tensor
carries a `comfy_quant` marker whose JSON this script records verbatim:

  - fp8: `{"format": "float8_e4m3fn"}`, a SCALAR `weight_scale`, and for most
    tensors an `input_scale` -- the activations are quantized too. `mlp.fc2`
    instead carries `"full_precision_matrix_mult": true` and no `input_scale`.
  - int8: `{"format": "int8_tensorwise", "convrot": true,
    "convrot_groupsize": N}`, a per-output-row `weight_scale` [out, 1], and a
    stored weight that is NOT the weight: it is `W @ H^T` in a Hadamard basis
    over groups of N along the input dimension.

**The rotation is why a naive cross-format comparison is meaningless.** The
int8 side has to go through `dequantize_int8_convrot_weight` before it is in
the same basis as anything else. `--self-test` fails if it does not matter.

## What this measures, and what it cannot

Stored-weight fidelity only. Per module: relative Frobenius delta and cosine
(float64 accumulation, chunked over output rows), max absolute error, and the
distribution of per-output-channel relative error -- the last one because a
scalar scale and a per-row scale differ exactly there and nowhere in a whole-
tensor norm.

Neither the fp8 `input_scale` path nor whatever int8 does with activations
at runtime is observable in any comparison of stored weights, and the fp8 numbers here are
what the format costs before a single activation is quantized. The controlled
runtime measurement is a fixed-input first-step forward with `unet_name` arms,
which is `docs/open_experiments.md` #22's method.

The AdaLN is deliberately excluded from every vs-bf16 number: the bf16 release
is unpruned, so grading a pruned AdaLN against it measures the rank-8 pruning
residual, already measured by `bench/analyze_adaln_pruning.py`, and mixing the
two would report pruning loss as quantization loss. The only AdaLN question
asked here is whether the two pruned files carry the same one.

## The qkv layout differs between the release and the repack

The bf16 release and the Comfy repack both store one `attn.qkv_proj.weight`
[3 * heads * head_dim, hidden] under the same name, and the rows are NOT in
the same order. The release interleaves per head -- `[head][q|k|v][head_dim]`
-- and the repack concatenates -- `[q|k|v][head][head_dim]`, which is what
`comfy/ldm/minimax/model.py` splits on. Compared as stored, a qkv weight reads
a relative delta of 1.40 with cosine ~0 against its own bf16 origin, in BOTH
quantizations equally; reordered, it reads the same ~0.9% as every other int8
linear.

`probe_qkv_layout()` runs before any reference comparison and refuses to
measure if the reordering is not the better reading, so the reordering is
never applied on faith. `head_dim` is read from the checkpoint's own
`attn.q_norm.weight`, not typed.

## Attribution: what a difference between the files can be

`shared_tensors` byte-compares every tensor the two files hold in common that
neither quantizes. If they are identical, every difference between the files
attributes to the 200 quantized linears alone; if they are not, the two builds
differ elsewhere too and nothing here is a clean format comparison. The record
says which.

## Self-test (`--self-test`)

Four deliberate violations, on real tensors, before any measurement is
believed:

  - a tensor compared against itself must read delta 0 and cosine 1;
  - a perturbation of known relative size must be measured back at that size;
  - the int8 weight WITHOUT un-rotation, compared against fp8, must read a
    delta far worse than the un-rotated one. A pass here means the un-rotation
    is load-bearing; this is the check `bench/analyze_ref_lora.py`'s `dequant`
    would fail, and its cross-format cosines should be read with that in mind;
  - dequantizing at the wrong `convrot_groupsize` must degrade the result, so
    the groupsize read from each marker is doing work.

Needs `comfy_kitchen` (the un-rotation) and torch on CPU; no GPU, no server.
Run it with ComfyUI's interpreter, not `uv run`.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from analyze_checkpoint_delta import header, read  # noqa: E402

sys.path.insert(0, str(_HERE.parents[2]))
from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    dequantize_int8_convrot_weight)

QUANT_MODS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
N_BLOCKS = 50
ROW_CHUNK = 4096


# ------------------------------------------------------------------ reading

def marker(path: str, hdr: dict, base: int, mod: str) -> dict | None:
    """The `comfy_quant` JSON of a module, or None if it carries none."""
    key = mod + ".comfy_quant"
    if key not in hdr:
        return None
    o0, o1 = hdr[key]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + o0)
        return json.loads(f.read(o1 - o0).decode())


def raw_bytes(path: str, hdr: dict, base: int, name: str) -> bytes:
    o0, o1 = hdr[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + o0)
        return f.read(o1 - o0)


def read_fp8(path: str, hdr: dict, base: int, name: str) -> np.ndarray:
    """`read()` cannot: numpy has no fp8. Decode through torch."""
    info = hdr[name]
    buf = bytearray(raw_bytes(path, hdr, base, name))
    t = torch.frombuffer(buf, dtype=torch.uint8).view(torch.float8_e4m3fn)
    return t.to(torch.float32).reshape(info["shape"]).numpy()


def weight_in_compute_space(path: str, hdr: dict, base: int, mod: str,
                            *, rotate_back: bool = True,
                            groupsize: int | None = None) -> np.ndarray:
    """The module's weight as the matmul sees it, whatever the file stores.

    `rotate_back=False` and a wrong `groupsize` exist for the self-test; no
    measurement path passes either.
    """
    name = mod + ".weight"
    dtype = hdr[name]["dtype"]
    conf = marker(path, hdr, base, mod) or {}
    if dtype == "F8_E4M3":
        w = read_fp8(path, hdr, base, name)
        return w * read(path, hdr, base, mod + ".weight_scale")
    if dtype == "I8":
        q = torch.from_numpy(read(path, hdr, base, name).astype(np.int8))
        scale = torch.from_numpy(read(path, hdr, base, mod + ".weight_scale"))
        if not conf.get("convrot"):
            return (q.float() * scale).numpy()
        gs = groupsize if groupsize is not None else int(conf["convrot_groupsize"])
        if not rotate_back:
            return (q.float() * scale).numpy()
        return dequantize_int8_convrot_weight(q, scale, gs).numpy()
    return read(path, hdr, base, name)


# ------------------------------------------------------------------ metrics

def stats(ref: np.ndarray, cand: np.ndarray) -> dict:
    """Chunked float64 comparison of two weights of identical shape.

    Chunked because these reach 1.5e8 elements: a whole-tensor float64 cast
    is 1.2 GB per side, and the reduction is what needs the precision, not
    the storage.
    """
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {cand.shape}")
    sr = sc = sd = dot = 0.0
    amax = 0.0
    row_rel: list[float] = []
    for i in range(0, ref.shape[0], ROW_CHUNK):
        a = ref[i:i + ROW_CHUNK].astype(np.float64)
        b = cand[i:i + ROW_CHUNK].astype(np.float64)
        d = a - b
        sr += float((a * a).sum())
        sc += float((b * b).sum())
        sd += float((d * d).sum())
        dot += float((a * b).sum())
        amax = max(amax, float(np.abs(d).max()))
        na = np.sqrt((a * a).sum(axis=1))
        nd = np.sqrt((d * d).sum(axis=1))
        row_rel.extend((nd / np.maximum(na, 1e-30)).tolist())
    rr = np.asarray(row_rel)
    return {
        "rel_delta": float(np.sqrt(sd) / max(np.sqrt(sr), 1e-30)),
        "cos": float(dot / max(np.sqrt(sr) * np.sqrt(sc), 1e-30)),
        "max_abs_err": amax,
        "ref_norm": float(np.sqrt(sr)),
        "row_rel_median": float(np.median(rr)),
        "row_rel_p95": float(np.percentile(rr, 95)),
        "row_rel_max": float(rr.max()),
    }


# ------------------------------------------------------- shared / reference

def shared_tensor_compare(a_path: str, a_h: dict, a_b: int,
                          b_path: str, b_h: dict, b_b: int) -> dict:
    """Byte-compare every tensor both files hold that neither quantizes."""
    quant = {k[:-len(".comfy_quant")] for k in a_h if k.endswith(".comfy_quant")}
    quant |= {k[:-len(".comfy_quant")] for k in b_h if k.endswith(".comfy_quant")}
    common = sorted(set(a_h) & set(b_h))
    checked, differ = 0, []
    for name in common:
        mod = name.rsplit(".", 1)[0]
        if mod in quant or name.endswith(".comfy_quant"):
            continue
        checked += 1
        if raw_bytes(a_path, a_h, a_b, name) != raw_bytes(b_path, b_h, b_b, name):
            differ.append(name)
    return {"checked": checked, "identical": len(differ) == 0,
            "differing": differ[:20], "n_differing": len(differ),
            "only_in_a": sorted(set(a_h) - set(b_h))[:10],
            "only_in_b": sorted(set(b_h) - set(a_h))[:10]}


class Reference:
    """The bf16 release, streamed by shard from its safetensors index."""

    def __init__(self, root: Path):
        idx = root / "model.safetensors.index.json"
        if not idx.exists():
            raise FileNotFoundError(f"no safetensors index under {root}")
        self.map = json.loads(idx.read_text())["weight_map"]
        self.root = root
        self._hdr: dict[str, tuple[dict, int]] = {}
        missing = [s for s in sorted(set(self.map.values()))
                   if not (root / s).exists()]
        if missing:
            raise FileNotFoundError(f"shards absent: {missing}")

    def has(self, name: str) -> bool:
        return name in self.map

    def get(self, name: str) -> np.ndarray:
        if name not in self.map:
            raise KeyError(f"{name} is not in the reference index")
        shard = self.map[name]
        if shard not in self._hdr:
            self._hdr[shard] = header(str(self.root / shard))
        hdr, base = self._hdr[shard]
        info = hdr[name]
        need = base + info["data_offsets"][1]
        size = (self.root / shard).stat().st_size
        if size < need:
            raise OSError(f"{shard} is {size} bytes, {name} needs {need}: "
                          "the shard is incomplete")
        return read(str(self.root / shard), hdr, base, name)


def head_dim(hdr: dict) -> int:
    """From the file's own per-head norm, so nothing is typed."""
    return int(hdr["blocks.0.attn.q_norm.weight"]["shape"][0])


def hf_to_comfy(name: str, w: np.ndarray, hd: int) -> np.ndarray:
    """The release's row order, rewritten as the repack's. Identity for
    everything that is not a fused qkv."""
    if not name.endswith("attn.qkv_proj.weight"):
        return w
    nh = w.shape[0] // (3 * hd)
    return w.reshape(nh, 3, hd, -1).transpose(1, 0, 2, 3).reshape(w.shape)


def probe_qkv_layout(ref: "Reference", path: str, hdr: dict, base: int,
                     mod: str) -> dict:
    """Measure both readings of one qkv weight before trusting either."""
    hd = head_dim(hdr)
    w = weight_in_compute_space(path, hdr, base, mod)
    raw = ref.get(mod + ".weight")
    as_stored = stats(raw, w)["rel_delta"]
    reordered = stats(hf_to_comfy(mod + ".weight", raw, hd), w)["rel_delta"]
    return {"module": mod, "head_dim": hd, "as_stored": as_stored,
            "reordered": reordered,
            "reordering_is_better": reordered < 0.5 * as_stored}


def format_floor(w: np.ndarray, gs: int) -> dict:
    """What each format costs on this weight when quantized ideally, here.

    A shipped file's error is the format's floor plus whatever its calibration
    added. Without this, "fp8 reads 2.65% because e4m3 has three mantissa
    bits" is a mechanism story; with it, the floor is a number beside the
    shipped one and the difference is the calibration.

    fp8: one scalar scale, amax over the tensor against e4m3's 448. int8: the
    shipped path exactly -- rotate, per-output-row amax against 127, round,
    and back through `dequantize_int8_convrot_weight`.
    """
    from comfy_kitchen.backends.eager.quantization import (
        _build_hadamard, _rotate_weight)
    t = torch.from_numpy(w.astype(np.float32))

    s8 = float(t.abs().max()) / 448.0
    fp8 = ((t / s8).to(torch.float8_e4m3fn).to(torch.float32) * s8).numpy()

    h = _build_hadamard(gs, dtype=torch.float32)
    rot = _rotate_weight(t, h, gs)
    row = (rot.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-30)
    q = torch.clamp(torch.round(rot / row), -127, 127).to(torch.int8)
    i8 = dequantize_int8_convrot_weight(q, row, gs).numpy()

    return {"fp8_ideal": stats(w, fp8)["rel_delta"],
            "int8_ideal": stats(w, i8)["rel_delta"]}


# ---------------------------------------------------------------- self-test

def self_test(files: dict[str, tuple[str, dict, int]]) -> dict:
    mod = "blocks.0.attn.out_proj"
    out: dict[str, Any] = {}
    fp8_path, fp8_h, fp8_b = files["fp8"]
    i8_path, i8_h, i8_b = files["int8"]

    w8 = weight_in_compute_space(fp8_path, fp8_h, fp8_b, mod)
    out["identity"] = stats(w8, w8.copy())
    ok_identity = out["identity"]["rel_delta"] == 0.0 and abs(out["identity"]["cos"] - 1) < 1e-12

    rng = np.random.default_rng(0)
    noise = rng.standard_normal(w8.shape).astype(np.float32)
    noise *= (0.01 * np.linalg.norm(w8.astype(np.float64))
              / np.linalg.norm(noise.astype(np.float64)))
    out["perturbation"] = stats(w8, w8 + noise)
    ok_pert = abs(out["perturbation"]["rel_delta"] - 0.01) < 1e-3

    conf = marker(i8_path, i8_h, i8_b, mod) or {}
    gs = int(conf["convrot_groupsize"])
    w_un = weight_in_compute_space(i8_path, i8_h, i8_b, mod)
    w_rot = weight_in_compute_space(i8_path, i8_h, i8_b, mod, rotate_back=False)
    out["unrotated_vs_rotated"] = {
        "groupsize": gs,
        "unrotated": stats(w8, w_un),
        "still_rotated": stats(w8, w_rot),
    }
    ok_rot = (out["unrotated_vs_rotated"]["still_rotated"]["rel_delta"]
              > 5 * out["unrotated_vs_rotated"]["unrotated"]["rel_delta"])

    wrong = 64 if gs != 64 else 256
    w_wrong = weight_in_compute_space(i8_path, i8_h, i8_b, mod, groupsize=wrong)
    out["wrong_groupsize"] = {"used": wrong, **stats(w8, w_wrong)}
    ok_gs = out["wrong_groupsize"]["rel_delta"] > 2 * out["unrotated_vs_rotated"]["unrotated"]["rel_delta"]

    out["passed"] = {
        "identity_reads_zero": ok_identity,
        "perturbation_measured_back": ok_pert,
        "unrotation_is_load_bearing": ok_rot,
        "wrong_groupsize_degrades": ok_gs,
    }
    out["all_passed"] = all(out["passed"].values())
    return out


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--models", type=Path, default=None,
                    help="ComfyUI diffusion_models dir; default resolves "
                         "../../models/diffusion_models from this repo")
    ap.add_argument("--checkpoint", default="fl2va", choices=("fl2va", "ref2va"))
    ap.add_argument("--reference", type=Path, default=None,
                    help="the bf16 release's transformer dir (with "
                         "model.safetensors.index.json). Omitted: the two "
                         "quantizations are compared only against each other")
    ap.add_argument("--blocks", default="all",
                    help="colon-separated block indices, or 'all'")
    ap.add_argument("--out", type=Path, default=None, help="write JSON here")
    ap.add_argument("--self-test", action="store_true",
                    help="run the deliberate violations and stop")
    args = ap.parse_args()

    models = args.models or (_HERE.parents[2] / "models" / "diffusion_models")
    paths = {
        "fp8": models / f"minimax_h3_{args.checkpoint}_pruned_fp8_scaled.safetensors",
        "int8": models / f"minimax_h3_{args.checkpoint}_pruned_int8_convrot.safetensors",
    }
    for tag, p in paths.items():
        if not p.exists():
            print(f"missing {tag}: {p}")
            return 2
    files = {tag: (str(p), *header(str(p))) for tag, p in paths.items()}

    if args.self_test:
        res = self_test(files)
        print(json.dumps(res, indent=2))
        return 0 if res["all_passed"] else 1

    blocks = (list(range(N_BLOCKS)) if args.blocks == "all"
              else [int(x) for x in args.blocks.split(":")])
    ref = Reference(args.reference) if args.reference else None
    hd = head_dim(files["int8"][1])

    rec: dict[str, Any] = {
        "date": date.today().isoformat(),
        "checkpoint": args.checkpoint,
        "what": ("fp8_scaled against int8_convrot on the same pruned "
                 "checkpoint, and each against the bf16 release when given: "
                 "stored-weight fidelity only"),
        "files": {t: p.name for t, p in paths.items()},
        # The last two components only: which release and which
        # component, without naming where this box keeps it.
        "reference": ("/".join(args.reference.resolve().parts[-2:])
                      if ref else None),
        "blocks": blocks,
        "not_measured": ("what either format does with activations at "
                         "run time: the fp8 input_scale path is stored and "
                         "int8's is not even visible here. The "
                         "controlled runtime version is open_experiments #22's "
                         "fixed-input first-step forward."),
        "adaln": ("excluded from every vs-bf16 number: the release is "
                  "unpruned, so that comparison would measure the rank-8 "
                  "pruning residual (analyze_adaln_pruning.py), not "
                  "quantization"),
    }

    print("== self-test")
    st = self_test(files)
    rec["self_test"] = st
    for k, v in st["passed"].items():
        print(f"   {'ok  ' if v else 'FAIL'} {k}")
    if not st["all_passed"]:
        print("self-test failed; refusing to measure")
        return 1

    print("== inventory")
    inv: dict[str, Any] = {}
    for tag, (p, h, b) in files.items():
        q = sorted({k[:-len(".comfy_quant")] for k in h if k.endswith(".comfy_quant")})
        markers: dict[str, int] = {}
        for m in q:
            markers[json.dumps(marker(p, h, b, m), sort_keys=True)] = \
                markers.get(json.dumps(marker(p, h, b, m), sort_keys=True), 0) + 1
        inv[tag] = {
            "n_tensors": len(h),
            "n_quantized": len(q),
            "n_input_scale": sum(1 for k in h if k.endswith(".input_scale")),
            "weight_scale_shape": h[q[0] + ".weight_scale"]["shape"],
            "markers": markers,
        }
        print(f"   {tag}: {len(h)} tensors, {len(q)} quantized, "
              f"{inv[tag]['n_input_scale']} input_scale, "
              f"weight_scale shape {inv[tag]['weight_scale_shape']}")
        for mk, n in markers.items():
            print(f"      {n:>4} x {mk}")
    rec["inventory"] = inv

    print("== shared (unquantized) tensors, byte-compared")
    rec["shared_tensors"] = shared_tensor_compare(*files["fp8"], *files["int8"])
    s = rec["shared_tensors"]
    print(f"   {s['checked']} compared, identical: {s['identical']}"
          + ("" if s["identical"] else f", {s['n_differing']} differ: {s['differing'][:5]}"))

    if ref:
        print("== qkv layout probe")
        probe = probe_qkv_layout(ref, *files["int8"], f"blocks.{blocks[0]}.attn.qkv_proj")
        rec["qkv_layout_probe"] = probe
        print(f"   as stored {probe['as_stored']:.5f}, reordered "
              f"{probe['reordered']:.5f}, head_dim {probe['head_dim']}")
        if not probe["reordering_is_better"]:
            print("   the release's qkv rows do not reorder into the repack's "
                  "layout; refusing to measure qkv against the reference")
            return 1

        print("== format floor, one module, quantized here from the release")
        fm = f"blocks.{blocks[0]}.mlp.fc1"
        gs = int((marker(*files["int8"], fm) or {})["convrot_groupsize"])
        floor = format_floor(hf_to_comfy(fm + ".weight", ref.get(fm + ".weight"), hd), gs)
        rec["format_floor"] = {"module": fm, "groupsize": gs, **floor}
        print(f"   {fm}: fp8 {floor['fp8_ideal']:.5f}, int8 {floor['int8_ideal']:.5f}")

    print("== per module")
    hdr_line = f"{'module':<28}{'fp8~int8':>10}{'cos':>9}"
    if ref:
        hdr_line += f"{'fp8~bf16':>10}{'int8~bf16':>11}{'i8 p95':>9}{'fp8 p95':>9}"
    print(hdr_line)
    rows = []
    for bi in blocks:
        for mod_suffix in QUANT_MODS:
            mod = f"blocks.{bi}.{mod_suffix}"
            w_fp8 = weight_in_compute_space(*files["fp8"], mod)
            w_i8 = weight_in_compute_space(*files["int8"], mod)
            row: dict[str, Any] = {
                "module": mod, "block": bi, "kind": mod_suffix,
                "fp8_vs_int8": stats(w_fp8, w_i8),
            }
            if ref:
                w_ref = hf_to_comfy(mod + ".weight", ref.get(mod + ".weight"), hd)
                row["fp8_vs_bf16"] = stats(w_ref, w_fp8)
                row["int8_vs_bf16"] = stats(w_ref, w_i8)
            rows.append(row)
            line = (f"{mod:<28}{row['fp8_vs_int8']['rel_delta']:>10.5f}"
                    f"{row['fp8_vs_int8']['cos']:>9.5f}")
            if ref:
                line += (f"{row['fp8_vs_bf16']['rel_delta']:>10.5f}"
                         f"{row['int8_vs_bf16']['rel_delta']:>11.5f}"
                         f"{row['int8_vs_bf16']['row_rel_p95']:>9.5f}"
                         f"{row['fp8_vs_bf16']['row_rel_p95']:>9.5f}")
            print(line)
    rec["modules"] = rows

    def summarise(key: str) -> dict | None:
        vals = [r[key]["rel_delta"] for r in rows if key in r]
        if not vals:
            return None
        by_kind = {}
        for k in QUANT_MODS:
            kv = [r[key]["rel_delta"] for r in rows if r["kind"] == k and key in r]
            by_kind[k] = {"median": float(np.median(kv)),
                          "min": float(np.min(kv)), "max": float(np.max(kv))}
        return {"median": float(np.median(vals)), "min": float(np.min(vals)),
                "max": float(np.max(vals)), "by_kind": by_kind}

    rec["summary"] = {k: summarise(k) for k in
                      ("fp8_vs_int8", "fp8_vs_bf16", "int8_vs_bf16")}
    print("== summary (relative Frobenius delta)")
    for k, v in rec["summary"].items():
        if v:
            print(f"   {k:<14} median {v['median']:.5f}  "
                  f"range {v['min']:.5f}-{v['max']:.5f}")

    if args.out:
        args.out.write_text(json.dumps(rec, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
