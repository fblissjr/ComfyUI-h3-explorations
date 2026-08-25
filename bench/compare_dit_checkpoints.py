"""Compare two H3 DiT safetensors files without downloading them.

Reads each file's header (local path or an `https://` URL via HTTP range
requests), diffs tensor names, dtypes, shapes and metadata, then fetches a
sample of tensors and compares bytes. Two hypotheses that a plain diff cannot
settle get their own tests:

- **qkv row order.** The release stores the fused `qkv_proj.weight` per head,
  `[q_h | k_h | v_h]`; ComfyUI's model splits it as `[q_all | k_all | v_all]`
  (`comfy/ldm/minimax/model.py`, `Attention.forward`) and nothing on its load
  path permutes. When the two files' qkv bytes differ, the test reads head 0's
  q, k and v rows from both under each layout and reports which layout each
  file is in. A file in release order is not ComfyUI-loadable as-is.
- **AdaLN factorisation.** Pruned files replace the timestep MLP with a curve
  table times a per-block linear. Two files can hold the same factorisation in
  different bases (sign flips, rotations) and different storage dtypes, so the
  tables and weights differ while the modulation does not. The test fetches
  the whole table and the named blocks' weight and bias from both, computes
  `table @ W.T + b` over the grid, and reports the relative difference, beside
  the least-squares residual of expressing one table in the other's basis.

Everything printed is MEASURED on the bytes fetched, and every sampled tensor
says how many bytes were read (a 2 MB cap per tensor; small tensors are read
whole). A byte-identical sample is evidence about that sample, not a proof of
the whole tensor; the record says which rows were compared.

Usage:

    python bench/compare_dit_checkpoints.py --a URL_OR_PATH --b URL_OR_PATH \\
        [--label-a dbm --label-b comfy] [--sample KEY ...] [--adaln blocks.0 final_layer] \\
        [--out bench/results/DATE_name.json]

With no `--sample`, a default set covers a norm weight, the patch projection,
the time embedder or curve table, the final layer, and three block weights
spread through the depth. `--adaln` runs the factorisation test on the named
blocks when both files carry `adaln_t_table`.

First run 2026-08-25 (record beside this file's results): the FL2VA files at
`DeepBeepMeep/MiniMax-H3` against `Comfy-Org/MiniMax-H3`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

SAMPLE_CAP = 2_000_000
DEFAULT_SAMPLES = (
    "blocks.0.norm1.weight",
    "video_patch_proj.weight",
    "time_embedder.proj_out.weight",
    "adaln_t_table",
    "final_layer.video_out.weight",
    "blocks.0.attn.qkv_proj.weight",
    "blocks.0.attn.qkv_proj.weight_scale",
    "blocks.0.attn.qkv_proj.comfy_quant",
    "blocks.25.attn.out_proj.weight",
    "blocks.49.mlp.fc1.weight",
)
NP_DTYPE = {"F32": np.float32, "F16": np.float16, "I8": np.int8, "U8": np.uint8,
            "F8_E4M3": np.uint8, "BF16": np.uint16}
WIDTH = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1, "F8_E4M3": 1, "I32": 4, "I64": 8}


class Source:
    """A safetensors file reachable by byte range: local path or https URL."""

    def __init__(self, ref: str):
        self.ref = ref
        self.remote = ref.startswith("http://") or ref.startswith("https://")
        self._fh = None if self.remote else open(ref, "rb")
        self.header_len = struct.unpack("<Q", self.read(0, 8))[0]
        self.header = json.loads(self.read(8, 8 + self.header_len))
        self.metadata = self.header.pop("__metadata__", None)
        self.bytes_read = 0

    def read(self, start: int, end: int) -> bytes:
        """Bytes [start, end)."""
        if end <= start:
            return b""
        if self.remote:
            req = urllib.request.Request(
                self.ref, headers={"Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
        else:
            assert self._fh is not None
            self._fh.seek(start)
            data = self._fh.read(end - start)
        if hasattr(self, "bytes_read"):
            self.bytes_read += len(data)
        return data

    def tensor(self, key: str, cap: int | None = SAMPLE_CAP):
        """(bytes, dtype, shape, whole?) for the first `cap` bytes of a tensor."""
        info = self.header[key]
        a, b = info["data_offsets"]
        base = 8 + self.header_len
        whole = cap is None or (b - a) <= cap
        stop = b if whole else a + cap
        return self.read(base + a, base + stop), info["dtype"], info["shape"], whole

    def rows(self, key: str, r0: int, r1: int) -> np.ndarray:
        """Rows [r0, r1) of a 2-D tensor as float32."""
        info = self.header[key]
        dt, shape = info["dtype"], info["shape"]
        rowb = shape[1] * WIDTH[dt]
        base = 8 + self.header_len + info["data_offsets"][0]
        raw = self.read(base + r0 * rowb, base + r1 * rowb)
        return as_float(raw, dt).reshape(r1 - r0, shape[1])

    def whole_float(self, key: str) -> np.ndarray:
        raw, dt, shape, _ = self.tensor(key, cap=None)
        return as_float(raw, dt).reshape(shape)


def as_float(raw: bytes, dt: str) -> np.ndarray:
    if dt == "BF16":
        return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dt in ("F32", "F16", "I8", "U8"):
        return np.frombuffer(raw, NP_DTYPE[dt]).astype(np.float32)
    raise ValueError(f"no float view for {dt}")


def header_diff(a: Source, b: Source) -> dict:
    ka, kb = set(a.header), set(b.header)
    shared = sorted(ka & kb)
    changed = []
    for k in shared:
        ia, ib = a.header[k], b.header[k]
        if ia["dtype"] != ib["dtype"] or ia["shape"] != ib["shape"]:
            changed.append({"key": k, "a": [ia["dtype"], ia["shape"]],
                            "b": [ib["dtype"], ib["shape"]]})
    def dtypes(s):
        out = {}
        for v in s.header.values():
            out[v["dtype"]] = out.get(v["dtype"], 0) + 1
        return out
    def tensor_bytes(s):
        return sum(int(np.prod(v["shape"])) * WIDTH.get(v["dtype"], 0)
                   for v in s.header.values())
    return {
        "tensors": [len(ka), len(kb)],
        "dtypes": [dtypes(a), dtypes(b)],
        "tensor_bytes": [tensor_bytes(a), tensor_bytes(b)],
        "header_bytes": [a.header_len, b.header_len],
        "only_in_a": sorted(ka - kb),
        "only_in_b": sorted(kb - ka),
        "dtype_or_shape_changed": changed,
        "metadata": [a.metadata, b.metadata],
    }


def sample_compare(a: Source, b: Source, keys) -> list[dict]:
    out = []
    for k in keys:
        if k not in a.header or k not in b.header:
            out.append({"key": k, "result": "absent in one file"})
            continue
        xa, da, sa, wa = a.tensor(k)
        xb, db, sb, _ = b.tensor(k)
        rec = {"key": k, "dtype": [da, db], "shape": sa, "bytes_compared": min(len(xa), len(xb)),
               "whole_tensor": wa}
        if da == db and sa == sb:
            rec["result"] = "identical" if xa == xb else "differ"
        elif da in ("F32", "F16", "BF16") and db in ("F32", "F16", "BF16"):
            fa, fb = as_float(xa, da), as_float(xb, db)
            m = min(len(fa), len(fb))
            fa, fb = fa[:m], fb[:m]
            rec["result"] = "dtype differs"
            rec["max_abs_diff"] = float(np.abs(fa - fb).max())
            rec["rel_diff"] = float(np.linalg.norm(fa - fb) / max(np.linalg.norm(fb), 1e-30))
        else:
            rec["result"] = "not comparable"
        out.append(rec)
    return out


def qkv_order(a: Source, b: Source, key="blocks.0.attn.qkv_proj.weight") -> dict | None:
    """Which fused-qkv layout each file holds, decided from head 0's rows."""
    if key not in a.header or key not in b.header:
        return None
    hd_key = "blocks.0.attn.q_norm.weight"
    dh = a.header[hd_key]["shape"][0]
    n_rows = a.header[key]["shape"][0]
    heads = n_rows // (3 * dh)
    def layout(src: Source):
        # Under release order rows [dh:2dh] are k_h0; under ComfyUI order they are q_h1.
        first = src.rows(key, 0, 3 * dh)
        k0_concat = src.rows(key, heads * dh, heads * dh + dh)
        v0_concat = src.rows(key, 2 * heads * dh, 2 * heads * dh + dh)
        return first, k0_concat, v0_concat
    fa, ka_c, va_c = layout(a)
    fb, kb_c, vb_c = layout(b)
    rec = {"key": key, "heads": heads, "head_dim": dh,
           "a_rows_0_3dh_equal_b": bool(np.array_equal(fa, fb))}
    if rec["a_rows_0_3dh_equal_b"]:
        rec["verdict"] = "same layout (head-0 rows identical)"
        return rec
    # Test a's rows [dh:3dh] against b's concatenated k0 / v0 and vice versa.
    a_is_release_b_concat = (np.array_equal(fa[dh:2 * dh], kb_c)
                             and np.array_equal(fa[2 * dh:3 * dh], vb_c))
    b_is_release_a_concat = (np.array_equal(fb[dh:2 * dh], ka_c)
                             and np.array_equal(fb[2 * dh:3 * dh], va_c))
    rec["a_release_order_b_comfy_order"] = bool(a_is_release_b_concat)
    rec["b_release_order_a_comfy_order"] = bool(b_is_release_a_concat)
    if a_is_release_b_concat:
        rec["verdict"] = "a is release order [head][q|k|v]; b is ComfyUI order [q|k|v]"
    elif b_is_release_a_concat:
        rec["verdict"] = "b is release order [head][q|k|v]; a is ComfyUI order [q|k|v]"
    else:
        rec["verdict"] = "rows differ and neither permutation explains it: values differ"
    return rec


def adaln_equivalence(a: Source, b: Source, blocks) -> dict | None:
    if "adaln_t_table" not in a.header or "adaln_t_table" not in b.header:
        return None
    ta, tb = a.whole_float("adaln_t_table"), b.whole_float("adaln_t_table")
    rec = {"table_shape": [list(ta.shape), list(tb.shape)], "blocks": {}}
    if ta.shape == tb.shape:
        rec["table_identical"] = bool(np.array_equal(ta, tb))
        r, *_ = np.linalg.lstsq(ta, tb, rcond=None)
        rec["basis_change_residual"] = float(np.linalg.norm(ta @ r - tb) / np.linalg.norm(tb))
    for blk in blocks:
        wk, bk = f"{blk}.adaln_proj.linear.weight", f"{blk}.adaln_proj.linear.bias"
        if wk not in a.header or wk not in b.header:
            rec["blocks"][blk] = "absent"
            continue
        wa, ba = a.whole_float(wk), a.whole_float(bk)
        wb, bb = b.whole_float(wk), b.whole_float(bk)
        ma, mb = ta @ wa.T + ba, tb @ wb.T + bb
        if ma.shape != mb.shape:
            # different grids: compare at matching fractions of the grid
            fr = np.linspace(0.0, 1.0, 9)
            ia = np.round(fr * (ma.shape[0] - 1)).astype(int)
            ib = np.round(fr * (mb.shape[0] - 1)).astype(int)
            rel = [float(np.linalg.norm(ma[i] - mb[j]) / np.linalg.norm(mb[j]))
                   for i, j in zip(ia, ib)]
            rec["blocks"][blk] = {"grids_differ": True, "rel_diff_at_9_fractions": rel}
        else:
            rec["blocks"][blk] = {
                "modulation_rel_diff": float(np.linalg.norm(ma - mb) / np.linalg.norm(mb)),
                "bias_rel_diff": float(np.linalg.norm(ba - bb) / np.linalg.norm(bb)),
                "weight_dtypes": [a.header[wk]["dtype"], b.header[wk]["dtype"]],
            }
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--sample", nargs="*", default=None,
                    help="tensor keys to byte-compare (default: a spread through the model)")
    ap.add_argument("--adaln", nargs="*", default=["blocks.0", "final_layer"],
                    help="blocks for the AdaLN factorisation test")
    ap.add_argument("--out", help="JSON record path")
    args = ap.parse_args()

    a, b = Source(args.a), Source(args.b)
    keys = args.sample if args.sample is not None else [
        k for k in DEFAULT_SAMPLES if k in a.header and k in b.header]
    rec = {
        "measured": _dt.date.today().isoformat(),
        "produced_by": "bench/compare_dit_checkpoints.py (header read + HTTP range samples)",
        # Local files are recorded by basename: an absolute path leaks the
        # machine's layout into a committed record and names a file that will
        # not exist elsewhere. URLs are recorded whole.
        "a": {"label": args.label_a, "ref": args.a if a.remote else Path(args.a).name},
        "b": {"label": args.label_b, "ref": args.b if b.remote else Path(args.b).name},
        "header_diff": header_diff(a, b),
        "samples": sample_compare(a, b, keys),
        "qkv_order": qkv_order(a, b),
        "adaln": adaln_equivalence(a, b, args.adaln),
        "bytes_read": [a.bytes_read, b.bytes_read],
    }

    hd = rec["header_diff"]
    print(f"{args.label_a} vs {args.label_b}: tensors {hd['tensors']}, "
          f"tensor bytes {[round(x / 1e9, 3) for x in hd['tensor_bytes']]} GB, "
          f"only-in-a {len(hd['only_in_a'])}, only-in-b {len(hd['only_in_b'])}, "
          f"dtype/shape changed {len(hd['dtype_or_shape_changed'])}")
    for s in rec["samples"]:
        extra = f"  rel {s['rel_diff']:.2e}" if "rel_diff" in s else ""
        print(f"  {s['key']:44} {str(s.get('dtype', '')):18} {s['result']:14}"
              f" ({s.get('bytes_compared', 0) // 1000} kB{', whole' if s.get('whole_tensor') else ''}){extra}")
    if rec["qkv_order"]:
        print("qkv order:", rec["qkv_order"]["verdict"])
    if rec["adaln"]:
        ad = rec["adaln"]
        print(f"adaln: tables {ad['table_shape']}"
              + (f", basis-change residual {ad['basis_change_residual']:.2e}"
                 if "basis_change_residual" in ad else ""))
        for blk, v in ad["blocks"].items():
            print(f"  {blk}: {v}")
    print(f"bytes read: {[round(x / 1e6, 1) for x in rec['bytes_read']]} MB")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rec, indent=2) + "\n")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
