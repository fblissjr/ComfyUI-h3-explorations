#!/usr/bin/env python3
"""Why block 49: per-head Q/K/V magnitudes on captured activations, joined to per-head error.

`bench/analyze_sol_error.py` found block 49's INT8 quantization error is
3-7x every other block's at the aggregate, while its per-head median is
ordinary (`bench/results/2026-08-19_sol_error_per_head.json`). The 2026-08-19
postmortem left "attribute the aggregate to named heads" open. This script
is that attribution, at the level the capture supports: the attention
INPUTS per head, not the outputs, which would need the kernel.

## What it measures

Per capture, per block, per head, for q, k and v: rms, max |x|, the largest
single-token norm, and the norm of the mean vector (what smooth-k removes).
Per head for q and k: how much of the head's energy sits in the channels
that dominate the block -- ranked by energy summed over heads -- and the
ratio of the loudest channel's rms to the median channel's, which is the
per-block INT8 quantiser's dynamic-range problem in one number.

From both checkpoints, for the same blocks: the `attn.q_norm` / `attn.k_norm`
RMSNorm gains -- rms, max, max/rms, top channels -- because a per-head rms
that differs across heads after an RMSNorm can only come from those gains
having outlier channels.

Where `--errors` names a per-head error record for the same capture, block
and step, every statistic is rank-correlated (Spearman) with that record's
`per_head_quant` and `per_head_sparsity`. The join refuses a record whose
`capture` field is not one of the captures given, and skips (loudly) a
block/step the record does not carry.

## What the 2026-08-20 run found (bench/results/2026-08-20_head_magnitudes.json)

Measured, on the 2026-08-17 (fl2va + ref LoRA) and 2026-08-18 (ref2va, no
LoRA) captures at step 3:

  - At block 49, per-head K rms runs from ~2.2 to ~32 and Q rms from ~2.2
    to ~11; at blocks 24 and 40 every head sits within a few percent of the
    same value. The heads with the largest K and Q rms are the heads with
    the largest INT8 error in the 2026-08-19 record (Spearman ~0.4-0.5
    against q_rms, k_rms and the loudest-channel ratio; ~0 against v).
  - Block 49's K energy is concentrated in four channels (82, 34, 67, 19
    carry ~93% of it summed over heads) where block 40's loudest channel
    carries ~1.4%. The same channels are where `attn.k_norm.weight` has its
    outliers: block 49's gains peak at ~37 and ~31 (channels 82 and 19)
    against an rms of ~5, where mid-depth blocks peak at ~1.9 against an rms
    of ~1.7. Blocks 45 and 48 carry one outlier channel each (~15 and ~11);
    they were not captured.
  - Both captures show the same structure to within a few percent, and the
    gains match between the two checkpoints to ~0.3%. The ref LoRA does not
    touch the norms. So the block 49 anomaly is a property of the released
    weights, present in fl2va and ref2va alike, and is NOT a fl2va-vs-ref2va
    differentiator. Inferred from that, not measured: a handful of heads
    whose Q and K live almost entirely in two or three very loud channels
    is the worst case for a per-block INT8 quantiser, which is the shape of
    the 2026-08-19 anomaly.
  - Missing control: no capture exists on clean fl2va. The fl2va arm here
    carries the rank-256 ref LoRA, which adapts attention and MLP linears.
    The inference survives because the gains are untouched by it and match
    across checkpoints, but "fl2va without a LoRA looks the same" is the
    one claim here that has not been observed.

Reads the capture tensors on CPU with mmap; no CUDA, no ComfyUI. The
capture paths are recorded as `$H3_CAPTURE_ROOT/<dir>`, not literally.

    python bench/analyze_head_magnitudes.py \\
        --capture $H3_CAPTURE_ROOT/2026-08-17_ref3_362f_1024x768 \\
        --capture $H3_CAPTURE_ROOT/2026-08-18_ref3_362f_1024x768_ref2va \\
        --blocks 49,40,24 --step 3 \\
        --errors bench/results/2026-08-19_sol_error_per_head.json \\
        --out bench/results/2026-08-20_head_magnitudes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_checkpoint_delta import FL_NAME, RE_NAME, header, read  # noqa: E402

N_HEADS = 56
TOP_CHANNELS = 4


def spearman(a, b) -> float:
    """Rank correlation without scipy, average ranks on ties."""
    def ranks(x):
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x)
        r = np.empty(len(x), dtype=np.float64)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def head_stats(t) -> dict:
    """Per-head statistics of one [heads, S, D] bf16 tensor, computed a head
    at a time so the mmap'd 1.4 GB tensor is never materialised at once."""
    import torch
    rms, maxabs, tokmax, meanvec = [], [], [], []
    chan_rms = np.zeros((t.shape[0], t.shape[-1]), dtype=np.float64)
    for h in range(t.shape[0]):
        x = t[h].float()
        rms.append(float(x.pow(2).mean().sqrt()))
        maxabs.append(float(x.abs().max()))
        tokmax.append(float(x.norm(dim=-1).max()))
        meanvec.append(float(x.mean(0).norm()))
        chan_rms[h] = x.pow(2).mean(0).sqrt().numpy()
    energy_by_channel = (chan_rms ** 2).sum(0)
    top = [int(c) for c in np.argsort(-energy_by_channel)[:TOP_CHANNELS]]
    per_head_energy = (chan_rms ** 2).sum(1)
    frac_top = ((chan_rms[:, top] ** 2).sum(1) / per_head_energy).tolist()
    loud_ratio = (chan_rms.max(1) / np.median(chan_rms, axis=1)).tolist()
    return {
        "rms": rms, "maxabs": maxabs, "token_norm_max": tokmax,
        "mean_vector_norm": meanvec,
        "top_channels_by_energy": top,
        "top_channels_energy_share": [round(float(energy_by_channel[c] / energy_by_channel.sum()), 4)
                                      for c in top],
        "per_head_energy_frac_in_top_channels": frac_top,
        "per_head_loudest_over_median_channel": loud_ratio,
    }


def gain_stats(path: str, blk: int) -> dict:
    h, b = header(path)
    out = {}
    for nm in ("attn.q_norm", "attn.k_norm"):
        g = read(path, h, b, f"blocks.{blk}.{nm}.weight").astype(np.float64)
        r = float(np.sqrt((g * g).mean()))
        top = np.argsort(-np.abs(g))[:3]
        out[nm] = {"rms": round(r, 4), "max_abs": round(float(np.abs(g).max()), 4),
                   "max_over_rms": round(float(np.abs(g).max() / r), 3),
                   "top_channels": [[int(i), round(float(g[i]), 3)] for i in top]}
    return out


def scrub(path: Path) -> str:
    return f"$H3_CAPTURE_ROOT/{path.name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", action="append", required=True,
                    help="capture directory; repeatable")
    ap.add_argument("--blocks", default="49,40,24")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--errors", default=None,
                    help="per-head error record from analyze_sol_error.py to join against")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--measured", default=date.today().isoformat())
    args = ap.parse_args()

    import torch
    caps = [Path(c).expanduser().resolve() for c in args.capture]
    for c in caps:
        if not c.is_dir():
            sys.exit(f"refuse: {c} is not a directory")
    blocks = [int(b) for b in args.blocks.split(",")]

    errors = None
    if args.errors:
        errors = json.load(open(args.errors))
        names = {c.name for c in caps}
        if errors.get("capture") not in names:
            sys.exit(f"refuse: --errors record is for capture {errors.get('capture')!r}, "
                     f"which is not among {sorted(names)}; a join across captures "
                     "would correlate heads from two different runs")
        for r in errors["rows"]:
            if r.get("heads_measured") != N_HEADS:
                sys.exit(f"refuse: --errors row b{r['block']} s{r['step']} measured "
                         f"{r.get('heads_measured')} heads, not {N_HEADS}; a prefix "
                         "cannot be joined head-for-head")
    err_rows = {(r["block"], r["step"]): r for r in (errors or {}).get("rows", [])}

    models = Path(args.models_dir) if args.models_dir else (
        HERE.parents[2] / "models" / "diffusion_models")

    out = {
        "measured": args.measured,
        "produced_by": "bench/analyze_head_magnitudes.py",
        "what": ("per-head q/k/v magnitude structure on captured activations, "
                 "q/k RMSNorm gain outliers from both checkpoints, joined to "
                 "per-head Sol error where a record exists"),
        "captures": [scrub(c) for c in caps],
        "step": args.step,
        "errors_record": args.errors,
        "gains": {}, "rows": [],
    }
    for blk in blocks:
        out["gains"][str(blk)] = {
            "fl2va": gain_stats(str(models / FL_NAME), blk),
            "ref2va": gain_stats(str(models / RE_NAME), blk)}

    for cap in caps:
        for blk in blocks:
            f = cap / f"qkv_L98498_S98498_b{blk}_s{args.step}.pt"
            if not f.exists():
                cands = sorted(cap.glob(f"qkv_*_b{blk}_s{args.step}.pt"))
                if len(cands) != 1:
                    sys.exit(f"refuse: {cap.name} has {len(cands)} files for b{blk} s{args.step}")
                f = cands[0]
            d = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
            row = {"capture": scrub(cap), "file": f.name, "block": blk, "step": args.step,
                   "seq_len": int(d["q"].shape[2])}
            for name in ("q", "k", "v"):
                t = d[name][0]
                if t.shape[0] != N_HEADS:
                    sys.exit(f"refuse: {f.name} {name} has {t.shape[0]} heads")
                row[name] = head_stats(t)
            er = err_rows.get((blk, args.step)) if (errors and errors["capture"] == cap.name) else None
            if errors and errors["capture"] == cap.name and er is None:
                print(f"  note  no error row for b{blk} s{args.step} in {args.errors}; join skipped",
                      flush=True)
            if er is not None:
                join = {}
                for target in ("per_head_quant", "per_head_sparsity"):
                    y = er[target]
                    join[target] = {
                        f"{name}_{stat}": round(spearman(row[name][stat], y), 3)
                        for name in ("q", "k", "v")
                        for stat in ("rms", "maxabs", "token_norm_max")}
                    for name in ("q", "k"):
                        join[target][f"{name}_energy_frac_in_top_channels"] = round(
                            spearman(row[name]["per_head_energy_frac_in_top_channels"], y), 3)
                        join[target][f"{name}_loudest_over_median_channel"] = round(
                            spearman(row[name]["per_head_loudest_over_median_channel"], y), 3)
                worst = np.argsort(-np.asarray(er["per_head_quant"]))[:8]
                join["worst_quant_heads"] = [
                    {"head": int(h), "quant": round(er["per_head_quant"][h], 4),
                     "q_rms": round(row["q"]["rms"][h], 3), "k_rms": round(row["k"]["rms"][h], 3),
                     "v_rms": round(row["v"]["rms"][h], 3)} for h in worst]
                row["join"] = join
            out["rows"].append(row)

            q, k, v = (np.asarray(row[n]["rms"]) for n in ("q", "k", "v"))
            print(f"{cap.name} b{blk} s{args.step}: q rms med {np.median(q):.2f} max {q.max():.2f} | "
                  f"k rms med {np.median(k):.2f} max {k.max():.2f} | v rms med {np.median(v):.1f} "
                  f"max {v.max():.1f} | k top channels {row['k']['top_channels_by_energy']} "
                  f"share {row['k']['top_channels_energy_share']}", flush=True)
            if er is not None:
                j = row["join"]["per_head_quant"]
                print(f"    spearman(quant, .): q_rms {j['q_rms']} k_rms {j['k_rms']} v_rms {j['v_rms']} "
                      f"k_loud {j['k_loudest_over_median_channel']} q_top {j['q_energy_frac_in_top_channels']}",
                      flush=True)

    Path(args.out).write_text(json.dumps(out, indent=1))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
