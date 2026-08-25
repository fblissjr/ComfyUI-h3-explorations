"""Compare H3 LoRA safetensors across naming conventions, by bytes.

The same LoRA circulates under three key conventions, and a filename says
nothing about which one a file holds or whether its tensors are the other
file's tensors:

- **diffusers**: `transformer_blocks.N.attn.to_q.lora_A.default.weight`,
  `ff.net.0.proj`, `ff.net.2`, `token_refiner.refiner_blocks.N...`; alpha, if
  anywhere, in `__metadata__`.
- **diffusers plus alpha tensors**: the same keys with one `.alpha` F32
  scalar per module added (what `DeepBeepMeep/MiniMax-H3/loras` ships).
- **ComfyUI native**: `diffusion_model.blocks.N.attn.qkv_proj.lora_A.weight`
  with the three attention LoRAs fused into one of triple rank, `mlp.fc1`
  fusing the SwiGLU pair, `.alpha` per module (lightx2v's `*_comfyui_bf16`).

This script reads headers (local path or `https://` by range request), maps
keys between conventions, byte-compares the mapped tensors it can (whole
tensors, capped per tensor), reads every alpha scalar, and reports what is
unmapped. For a fused ComfyUI tensor it checks the block structure: the fused
`lora_A` should stack the three (or two) source `lora_A`s and the fused
`lora_B` should place each source `lora_B` on its own column band.

MEASURED means bytes fetched; nothing here loads a model. First run
2026-08-25 across `lightx2v/Minimax-h3-Turbo` and `DeepBeepMeep/MiniMax-H3`.

Usage:

    python bench/compare_lora_files.py --a URL_OR_PATH --b URL_OR_PATH \\
        [--label-a x --label-b y] [--samples 12] [--out bench/results/DATE_name.json]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_dit_checkpoints import Source, as_float  # noqa: E402

CAP = 20_000_000

# diffusers module path -> ComfyUI module path (and which fused band it lands in)
_D2C = [
    (r"^transformer_blocks\.(\d+)\.attn\.to_q$", r"blocks.\1.attn.qkv_proj", 0),
    (r"^transformer_blocks\.(\d+)\.attn\.to_k$", r"blocks.\1.attn.qkv_proj", 1),
    (r"^transformer_blocks\.(\d+)\.attn\.to_v$", r"blocks.\1.attn.qkv_proj", 2),
    (r"^transformer_blocks\.(\d+)\.attn\.to_out\.0$", r"blocks.\1.attn.out_proj", None),
    (r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj$", r"blocks.\1.mlp.fc1", 0),
    (r"^transformer_blocks\.(\d+)\.ff\.net\.2$", r"blocks.\1.mlp.fc2", None),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_q$", r"token_refiner.blocks.\1.attn.qkv_proj", 0),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_k$", r"token_refiner.blocks.\1.attn.qkv_proj", 1),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_v$", r"token_refiner.blocks.\1.attn.qkv_proj", 2),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_out\.0$", r"token_refiner.blocks.\1.attn.out_proj", None),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.0\.proj$", r"token_refiner.blocks.\1.mlp.fc1", 0),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.2$", r"token_refiner.blocks.\1.mlp.fc2", None),
]


def split_key(key: str):
    """(module, part) where part is 'lora_A' | 'lora_B' | 'alpha' | None."""
    key = re.sub(r"^diffusion_model\.", "", key)
    m = re.match(r"^(.*)\.(lora_A|lora_B|lora_down|lora_up)(?:\.default)?\.weight$", key)
    if m:
        part = {"lora_down": "lora_A", "lora_up": "lora_B"}.get(m.group(2), m.group(2))
        return m.group(1), part
    m = re.match(r"^(.*)\.alpha$", key)
    if m:
        return m.group(1), "alpha"
    return key, None


def to_comfy_module(module: str):
    """diffusers module -> (comfy module, band) or (module, None) if already comfy."""
    for pat, rep, band in _D2C:
        if re.match(pat, module):
            return re.sub(pat, rep, module), band
    return module, None


def index(src: Source):
    """{module: {part: key}} in ComfyUI module names, with band info."""
    out = {}
    for k in src.header:
        module, part = split_key(k)
        if part is None:
            out.setdefault(("__other__", k), {})[None] = k
            continue
        cm, band = to_comfy_module(module)
        out.setdefault(cm, {}).setdefault(part, {})[band] = k
    return out


def load(src: Source, key: str):
    raw, dt, shape, whole = src.tensor(key, cap=CAP)
    return as_float(raw, dt).reshape(-1) if whole else as_float(raw, dt), dt, shape, whole


def compare_pair(a: Source, b: Source, n_samples: int):
    ia, ib = index(a), index(b)
    mods_a = {m for m in ia if not (isinstance(m, tuple))}
    mods_b = {m for m in ib if not (isinstance(m, tuple))}
    shared = sorted(mods_a & mods_b)
    rec = {"modules": [len(mods_a), len(mods_b), len(shared)],
           "only_in_a": sorted(mods_a - mods_b)[:10], "only_in_b": sorted(mods_b - mods_a)[:10],
           "alphas": {}, "samples": []}

    # alphas: every scalar, whole
    for label, src, idx in (("a", a, ia), ("b", b, ib)):
        vals = {}
        for m, parts in idx.items():
            if isinstance(m, tuple) or "alpha" not in parts:
                continue
            for band, k in parts["alpha"].items():
                v, *_ = load(src, k)
                vals.setdefault(float(v[0]), 0)
                vals[float(v[0])] += 1
        meta_alpha = (src.metadata or {}).get("alpha") or (src.metadata or {}).get("training_alpha")
        rec["alphas"][label] = {"tensor_values": vals, "metadata_alpha": meta_alpha}

    # pick modules spread through the list, prefer ones present in both with A and B
    cand = [m for m in shared if "lora_A" in ia[m] and "lora_A" in ib[m]]
    # spread the samples over module kinds (qkv, out_proj, fc1, fc2, adaln...),
    # otherwise an alphabetical stride never reaches the fused attention tensors
    by_kind = {}
    for m in cand:
        by_kind.setdefault(m.rsplit(".", 1)[-1] if not m.endswith("linear") else "adaln", []).append(m)
    picked = []
    per_kind = max(1, n_samples // max(1, len(by_kind)))
    for kind, ms in sorted(by_kind.items()):
        step = max(1, len(ms) // per_kind)
        picked.extend(ms[::step][:per_kind])
    for m in picked:
        for part in ("lora_A", "lora_B"):
            pa, pb = ia[m].get(part, {}), ib[m].get(part, {})
            if not pa or not pb:
                continue
            s = {"module": m, "part": part}
            if set(pa) == set(pb):
                # same convention: compare band by band (usually just None)
                results = []
                for band in pa:
                    xa, da, sa, wa = load(a, pa[band]); xb, db, sb, wb = load(b, pb[band])
                    n = min(len(xa), len(xb))
                    results.append({"band": band, "dtype": [da, db], "shape": sa, "whole": wa and wb,
                                    "identical": bool(np.array_equal(xa[:n], xb[:n])),
                                    "rel_diff": float(np.linalg.norm(xa[:n] - xb[:n]) / max(np.linalg.norm(xb[:n]), 1e-30))})
                s["results"] = results
            else:
                # one side fused (single key, band None), other side banded
                fused_src, fused_key, banded_src, banded = (
                    (a, pa[None], b, pb) if None in pa and None not in pb else (b, pb[None], a, pa))
                s["fused_side"] = "a" if fused_src is a else "b"
                xf, df, sf, wf = load(fused_src, fused_key)
                if not wf:
                    s["results"] = "fused tensor larger than cap; skipped"
                    rec["samples"].append(s); continue
                F = xf.reshape(sf)
                bands = []
                for band in sorted(banded):
                    xb_, db_, sb_, wb_ = load(banded_src, banded[band])
                    if not wb_:
                        bands.append({"band": band, "skipped": "over cap"}); continue
                    B = xb_.reshape(sb_)
                    if part == "lora_A":      # fused A stacks source As along rows
                        r = B.shape[0]; seg = F[band * r:(band + 1) * r, :]
                    else:                      # fused B places source B on its own column band
                        r = B.shape[1]; seg = F[:, band * r:(band + 1) * r] if F.shape[0] == B.shape[0] else None
                        if seg is None:        # fc1 SwiGLU: rows stacked instead
                            n0 = B.shape[0]; seg = F[band * n0:(band + 1) * n0, band * r:(band + 1) * r]
                    ok = seg is not None and seg.shape == B.shape
                    entry = {"band": band, "shape_src": sb_, "shape_fused": sf,
                             "identical": bool(ok and np.array_equal(seg, B)),
                             "rel_diff": float(np.linalg.norm(seg - B) / max(np.linalg.norm(B), 1e-30)) if ok else None}
                    if ok and not entry["identical"] and part == "lora_B" and seg.shape == F.shape:
                        # same-shape mismatch on an output-row tensor: test the row permutations
                        # a SwiGLU gate/up re-layout would produce
                        n = F.shape[0]; h = n // 2
                        perms = {"halves_swapped": np.concatenate([F[h:], F[:h]]),
                                 "deinterleave_even_odd": np.concatenate([F[0::2], F[1::2]]),
                                 "interleave_halves": np.stack([F[:h], F[h:]], 1).reshape(n, -1)}
                        entry["permutation_match"] = next((nm for nm, P in perms.items() if np.array_equal(P, B)), None)
                    bands.append(entry)
                s["results"] = bands
            rec["samples"].append(s)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="a"); ap.add_argument("--label-b", default="b")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--out")
    args = ap.parse_args()
    a, b = Source(args.a), Source(args.b)
    rec = {"measured": _dt.date.today().isoformat(),
           "produced_by": "bench/compare_lora_files.py (header read + range-fetched tensors)",
           "a": {"label": args.label_a, "ref": args.a if a.remote else Path(args.a).name,
                 "tensors": len(a.header), "metadata": a.metadata},
           "b": {"label": args.label_b, "ref": args.b if b.remote else Path(args.b).name,
                 "tensors": len(b.header), "metadata": b.metadata}}
    rec.update(compare_pair(a, b, args.samples))
    rec["bytes_read"] = [a.bytes_read, b.bytes_read]

    print(f"{args.label_a} vs {args.label_b}: modules {rec['modules']} (a, b, shared); "
          f"only-in-a {len(rec['only_in_a'])} only-in-b {len(rec['only_in_b'])}")
    for side, src in (("a", a), ("b", b)):
        m = src.metadata or {}
        keys = {k: m[k] for k in ("swi_glu_mapping", "target_format", "source_format", "key_format", "training_rank", "source") if k in m}
        if keys: print(f"  metadata {side}: {keys}")
        al = rec["alphas"][side]
        print(f"  alpha {side}: tensors {al['tensor_values'] or 'none'}; metadata {al['metadata_alpha']}")
    ident = differ = 0
    for s in rec["samples"]:
        res = s["results"]
        if isinstance(res, str):
            print(f"  {s['module']:40} {s['part']}: {res}"); continue
        flags = []
        for r in res:
            if r.get("skipped"): flags.append("skip"); continue
            ok = r["identical"]; ident += ok; differ += (not ok)
            tag = "=" if ok else (f"perm:{r['permutation_match']}" if r.get("permutation_match") else f"~{r['rel_diff']:.1e}")
            flags.append(tag + (f"[{r['band']}]" if r.get('band') is not None else ""))
        print(f"  {s['module']:40} {s['part']}: {' '.join(flags)}")
    print(f"  identical {ident}, differing {differ}; bytes read {[round(x/1e6,1) for x in rec['bytes_read']]} MB")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rec, indent=2) + "\n"); print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
