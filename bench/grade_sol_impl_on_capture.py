#!/usr/bin/env python3
"""Grade our Sol node's implementation against ComfyUI core's, on the SAME captured inputs.

Pair B of `bench/sol_core_ab_arms.json` (retired 2026-09-15; its rendered arms
are listed in `data/results/2026-09-10_sol_core_ab_outputs.json`) renders our
node re-set to core's knob values against core's own `BlockSparseAttention`
(Comfy-Org/ComfyUI#16072).
The two then differ in implementation, not policy, and a rendered clip cannot
attribute a numerical change (CLAUDE.md). This does, at the call level:

  ours   `ck.sol_attn` on the normed, roped bf16 Q/K/V ComfyUI's H3 forward
         builds (fused `rms_rope_split_half_` in place on the qkv buffer),
         with our node's sink ranges -- exactly `sol_attn_h3._run` and
         `sol_attn_h3._sink_blocks`, imported, not copied.
  core   `ck.sol_attn_chunked` fed the `qkv_proj` output in core's
         `PRODUCER_CHUNK`-row chunks, q/k RMSNorm and RoPE applied inside the
         kernel, with core's sink ranges (`SparseAttnPatch.sinks`, imported)
         and the previous step's K-mean / V-scale carried as core's node
         carries them.

Inputs are `qkvpre_*.pt` records from `h3_capture.py` with `pre=` armed: the
fused projection before RMSNorm and RoPE, the rope table, the q/k norm
weights and eps, and the packed layout's segment table. Capture on a graph
with Sol ABSENT so every cell sits on the dense-sage trajectory; the Base16
capture's graph, `workflows/bench/h3_text_to_video_stamped_api.json`, is one.

Per cell, every arm is scored against fp32 exact attention on the same Q/K/V,
rebuilt here with the same kitchen op the forward runs:

  control_ours_all_routed  ck.sol_attn at tau -1e9 (every block exact): the
                           instrument's floor for our path.
  control_core_all_routed  ck.sol_attn_chunked at tau -1e9, fresh statistics:
                           the same floor for core's producer, so a gap at the
                           matched tau is routing and quantisation, not a
                           producer defect.
  ours                     our call at the matched knobs.
  ours_core_sinks          only when the two sink derivations disagree on the
                           cell: ours with core's ranges, which separates the
                           derivation from the implementation.
  core_fresh               core's call with kmean/vscale None (measured from
                           this step's own K and V).
  core_carried             core's call with kmean/vscale from core's call on
                           the same block at step-1, as its node carries them
                           (`pooled` in `h3_sparse_attention`). Only when
                           step-1 of the same block and branch was captured.
                           kitchen harvests those statistics from the step's
                           own K and V whatever scales it was handed (its
                           bootstrap comment), so the step-1 fresh call gives
                           what core's chain carries.
  ours_shipped             ours at the shipped per-call values
                           (`h3_config.SOL_RECOMMENDED_CUDA`), which ties the
                           grade to Pair A's "ours as shipped" arm.

Metrics, per arm against exact: whole-tensor relative L2 and cosine, per-head
cosine (mean, worst, list), per-row relative L2 and cosine -- the functions of
`bench/measure_sol_exact_variants.py`, imported -- plus the same two numbers
per segment kind of the layout (text, audio, video, ...), and core-against-ours
directly. Each arm's single-call time is recorded with its note; it is not a
render time.

Controls on the instrument itself:
  reproduction  when the capture also holds the post-RoPE `qkv_*` file of a
                cell (`pre=1`), the rebuilt Q/K/V are compared to it; they
                should be identical, and if not every arm is suspect.
  chunk_check   read from the capture: whether projecting core's first chunk
                alone reproduced those rows of the full projection. The core
                arm is fed chunks of the full projection, which equals core's
                own chunked projection only when that holds.

Matched knob values default to `h3_config.SOL_CORE_DEFAULTS` (tau, extra_tokens
as token_aug, sink mode). Needs the card with nothing resident: a cell holds
the fused projection, a rebuilt copy, an fp32 reference and one arm output at
a time. Writes `data/results/<date>_sol_impl_capture_grade.json`.

    <comfy-venv-python> bench/grade_sol_impl_on_capture.py --capture <dir> --limit 2   # first run
    <comfy-venv-python> bench/grade_sol_impl_on_capture.py --capture <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent

_PRE = re.compile(r"^qkvpre_L[^_]+_S(\d+)_b(\d+)_s(\d+)(?:_r(\d+))?\.pt$")
_POST = re.compile(r"^qkv_L[^_]+_S(\d+)_b(\d+)_s(\d+)(?:_k[a-z0-9]+)?(?:_r(\d+))?\.pt$")


def _load_pack_module(name):
    """Load one module of this pack under a namespace package, the way
    `bench/check_sol_node_equivalence.py` does on CPU, so the whole pack's
    node registration is not imported to reach two functions."""
    import importlib.util
    if "h3x" not in sys.modules:
        pkg = types.ModuleType("h3x")
        pkg.__path__ = [str(REPO)]
        sys.modules["h3x"] = pkg
    full = f"h3x.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, REPO / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _git_head(root: Path):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                  # noqa: BLE001 -- unknown is a value
        return None


def segment_stats(a, b, segments):
    """Relative L2 and cosine of `a` against reference `b` (both BTHD) over the
    rows of each segment KIND, accumulated per head so no whole-tensor fp32
    copy is made. None when the capture carried no segment table."""
    if not segments:
        return None
    out = {}
    for kind in sorted({k for _, _, k in segments}):
        diff2 = ref2 = dot = a2 = 0.0
        rows = 0
        for s0, s1, k in segments:
            if k != kind or s1 <= s0:
                continue
            rows += s1 - s0
            for hi in range(a.shape[2]):
                x = a[:, s0:s1, hi].float().flatten()
                y = b[:, s0:s1, hi].float().flatten()
                diff2 += float(((x - y) ** 2).sum())
                ref2 += float((y ** 2).sum())
                dot += float(x @ y)
                a2 += float((x ** 2).sum())
        if rows and ref2 > 0 and a2 > 0:
            out[kind] = {"rows": rows, "rel_l2": math.sqrt(diff2 / ref2), "cos": dot / math.sqrt(a2 * ref2)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--capture", required=True, help="directory of h3_capture.py qkvpre_*.pt records")
    ap.add_argument("--out", default=None, help="record path (default data/results/<date>_sol_impl_capture_grade.json)")
    ap.add_argument("--limit", type=int, default=None, help="grade only the first K cells (a first run, not a record)")
    ap.add_argument("--chunk", type=int, default=2048, help="query rows per fp32 dense chunk")
    ap.add_argument("--tau", type=float, default=None, help="matched tau (default h3_config.SOL_CORE_DEFAULTS)")
    ap.add_argument("--token-aug", type=int, default=None, help="matched token_aug (default SOL_CORE_DEFAULTS extra_tokens)")
    ap.add_argument("--sink", default=None, help="matched sink mode (default SOL_CORE_DEFAULTS sink_conditioning)")
    ap.add_argument("--no-shipped", dest="shipped", action="store_false", help="skip the ours_shipped arm")
    ap.add_argument("--reproduce-all", action="store_true",
                    help="run the reproduction control on every cell with a post-RoPE twin, not only the first")
    ap.add_argument("--kernel-source", default=None, metavar="TEXT",
                    help="where the graded build came from, recorded verbatim beside the version")
    args = ap.parse_args()

    sys.path.insert(0, str(COMFY))
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO / "workflows"))
    try:
        import importlib.metadata as md
        import torch
        import comfy.quant_ops
        ck = comfy.quant_ops.ck
        import comfy_extras.nodes_sparse_attention as core_sparse
        import measure_sol_exact_variants as msev
        from h3_config import SOL_CORE_DEFAULTS, SOL_RECOMMENDED_CUDA
        ours_mod = _load_pack_module("sol_attn_h3")
    except Exception as exc:                          # noqa: BLE001
        print(f"SKIP: {type(exc).__name__}: {exc}")
        return 2
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 2

    tau = args.tau if args.tau is not None else float(SOL_CORE_DEFAULTS["selection.tau"])
    aug = args.token_aug if args.token_aug is not None else int(SOL_CORE_DEFAULTS["extra_tokens"])
    sink_mode = args.sink or SOL_CORE_DEFAULTS["sink_conditioning"]
    min_tokens = int(SOL_CORE_DEFAULTS["min_tokens"])
    shipped = dict(tau=float(SOL_RECOMMENDED_CUDA["tau"]), token_aug=0,
                   sink=SOL_RECOMMENDED_CUDA["sink_conditioning"])
    chunk_rows = int(core_sparse.PRODUCER_CHUNK)

    cap = Path(os.path.expanduser(args.capture))
    cells = []
    for f in sorted(glob.glob(str(cap / "qkvpre_*.pt"))):
        m = _PRE.match(os.path.basename(f))
        if m:
            cells.append((int(m.group(4) or 0), int(m.group(2)), int(m.group(3)), f))
    if not cells:
        print(f"no qkvpre_*.pt under {args.capture}; capture with H3_CAPTURE=...,pre=1")
        return 1
    cells.sort()
    if args.limit:
        cells = cells[:args.limit]
    twins = {}
    for f in glob.glob(str(cap / "qkv_*.pt")):
        m = _POST.match(os.path.basename(f))
        if m:
            twins[(int(m.group(4) or 0), int(m.group(2)), int(m.group(3)))] = f

    manifest = None
    if (cap / "manifest.json").is_file():
        try:
            mj = json.loads((cap / "manifest.json").read_text())
            manifest = {"timestamp": mj.get("timestamp"), "schema_version": mj.get("schema_version"),
                        "graph_sha256": (mj.get("workload") or {}).get("graph_sha256")}
        except Exception as exc:                      # noqa: BLE001
            manifest = {"error": str(exc)}

    rec = {"produced_by": "bench/grade_sol_impl_on_capture.py",
           "question": "at matched knob values, how far is ComfyUI core's BlockSparseAttention H3 path "
                       "(chunked producer, in-kernel norm+RoPE, carried K/V statistics) from ours, and each "
                       "from exact attention, on identical captured inputs",
           "comfy_kitchen": md.version("comfy_kitchen"), "kernel_source": args.kernel_source,
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "repo_commit": _git_head(REPO), "comfy_commit": _git_head(COMFY),
           "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "capture": {"dir": cap.name, "cells": len(cells), "limited_to_first": args.limit,
                       "manifest": manifest},
           "knobs": {"matched": {"tau": tau, "token_aug": aug, "sink_conditioning": sink_mode,
                                 "tail": True, "min_tokens": min_tokens,
                                 "source": "h3_config.SOL_CORE_DEFAULTS unless overridden on the command line"},
                     "shipped": shipped, "producer_chunk": chunk_rows,
                     "reference": "fp32 chunked softmax attention, per head"},
           "cells": []}
    print(f"grading {len(cells)} cells on {rec['comfy_kitchen']}: matched tau {tau} token_aug {aug} "
          f"sink {sink_mode}; producer chunk {chunk_rows}\n")

    def timed(fn):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize()
        return out, s.elapsed_time(e)

    def score(out, ref, segments):
        r, c = msev.rel_cos_lean(out, ref)
        ph = msev.per_head_cos(out, ref)
        return {"rel_l2": r, "cos": c, "cos_mean": sum(ph) / len(ph), "cos_worst": min(ph),
                "per_head": ph, **msev.row_stats(out, ref), "segments": segment_stats(out, ref, segments)}

    carry = {}   # (render, block, branch) -> (step, kmean_next, vscale_next)
    reproduced_once = False
    with torch.inference_mode():
        for render, block, step, path in cells:
            d = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            heads, hd = int(d["heads"]), int(d["head_dim"])
            qkv = d["qkv"].to("cuda", torch.bfloat16).contiguous()
            s_len = int(qkv.shape[0])
            rope = d["rope_freqs"].to("cuda")
            qw = d["q_norm_weight"].to("cuda")
            kw = d["k_norm_weight"].to("cuda")
            eps, rot = float(d["rope_eps"]), int(d["rot_dim"])
            segments = [tuple(x) for x in (d.get("segments") or [])] or None
            branch = json.dumps(d.get("uuids"), default=str)
            row = {"file": os.path.basename(path), "render": render, "block": block, "step": step,
                   "sigma": d.get("sigma"), "seq_len": s_len, "heads": heads,
                   "chunk_check": d.get("chunk_check"), "segments": segments,
                   "cond_or_uncond": d.get("cond_or_uncond")}

            # Rebuild what ComfyUI's forward hands attention: the same op, in place,
            # on a copy of the captured buffer, as three views of it.
            def rebuild():
                buf = qkv.clone()
                q, k, v = buf.split(heads * hd, dim=-1)
                q = q.view(1, s_len, heads, hd)
                k = k.view(1, s_len, heads, hd)
                v = v.view(1, s_len, heads, hd)
                ck.rms_rope_split_half_(q, k, rope, qw, kw, epsilon=eps, rot_dim=rot)
                return q, k, v
            (q, k, v), prep_ms = timed(rebuild)
            row["prep_ms"] = prep_ms

            twin = twins.get((render, block, step))
            if twin and (args.reproduce_all or not reproduced_once):
                t = torch.load(twin, map_location="cpu", weights_only=True, mmap=True)
                same = {}
                for n, mine in (("q", q), ("k", k), ("v", v)):
                    theirs = t[n].to("cuda").permute(0, 2, 1, 3)   # file is BHND
                    same[n] = {"identical": bool(torch.equal(mine, theirs)),
                               "max_abs_diff": float((mine.float() - theirs.float()).abs().max())}
                    del theirs
                row["reproduction"] = {"twin": os.path.basename(twin), **same}
                reproduced_once = True
                del t

            ref = msev.dense_fp32_chunked(q, k, v, args.chunk)

            # sink ranges, each node's own derivation
            video = next(((a, b) for a, b, kind in (segments or []) if kind == "video"), None)
            audio = next(((a, b) for a, b, kind in (segments or []) if kind == "audio"), None)
            ours_sinks = ours_mod.sink_ranges(video, audio, s_len, sink_mode)
            patch = core_sparse.SparseAttnPatch(
                tau=tau, topk_ratio=0.0, vsa=False, sigma_start=float("inf"), sigma_end=float("-inf"),
                min_tokens=min_tokens, dense_blocks=set(), sink_conditioning=sink_mode,
                extra_tokens=aug, verbose=False)
            layout = types.SimpleNamespace(segments=segments or [], seq_len=s_len)
            core_sinks = patch.sinks({"minimax_h3_layout": layout}, s_len)
            row["sinks"] = {"ours": [list(x) for x in ours_sinks], "core": [list(x) for x in core_sinks],
                            "match": tuple(map(tuple, ours_sinks)) == tuple(map(tuple, core_sinks))}

            bhnd = tuple(t_[0].transpose(0, 1).unsqueeze(0) for t_ in (q, k, v))

            def ours_call(t_, a_, sinks):
                out = ours_mod._run(*bhnd, heads, True, True, None, t_, min_tokens, False,
                                    sink_blocks=sinks[0], sink_q=sinks[1], topk_ratio=0.0,
                                    tail=True, token_aug=a_)
                if out is None:
                    raise RuntimeError("our node declined the call (below min_tokens or ineligible)")
                return out.transpose(1, 2)   # BHND -> BTHD

            def core_call(t_, a_, sinks, kmean=None, vscale=None):
                chunks = [qkv[i:i + chunk_rows] for i in range(0, s_len, chunk_rows)]
                return ck.sol_attn_chunked(chunks, s_len, heads, rope, (qw, kw), kmean=kmean, vscale=vscale,
                                           tau=t_, topk_ratio=0.0, token_aug=a_,
                                           sink_blocks=list(sinks[0]), sink_q=list(sinks[1]), rope_eps=eps)

            arms = {}
            out, ms = timed(lambda: ck.sol_attn(q, k, v, tau=-1e9))
            arms["control_ours_all_routed"] = {**score(out, ref, segments), "ms": ms}
            del out
            (out, _km, _vs), ms = timed(lambda: core_call(-1e9, 0, ((0, 0), (0, 0))))
            arms["control_core_all_routed"] = {**score(out, ref, segments), "ms": ms}
            del out, _km, _vs

            ours_out, ms = timed(lambda: ours_call(tau, aug, ours_sinks))
            arms["ours"] = {**score(ours_out, ref, segments), "ms": ms}
            if not row["sinks"]["match"]:
                out, ms = timed(lambda: ours_call(tau, aug, core_sinks))
                arms["ours_core_sinks"] = {**score(out, ref, segments), "ms": ms}
                del out

            (core_out, km_next, vs_next), ms = timed(lambda: core_call(tau, aug, core_sinks))
            arms["core_fresh"] = {**score(core_out, ref, segments), "ms": ms}
            r, c = msev.rel_cos_lean(core_out, ours_out)
            arms["core_fresh"]["vs_ours"] = {"rel_l2": r, "cos": c,
                                             "segments": segment_stats(core_out, ours_out, segments)}
            del core_out

            prev = carry.get((render, block, branch))
            if prev is not None and prev[0] == step - 1:
                (out, _km, _vs), ms = timed(lambda: core_call(tau, aug, core_sinks, kmean=prev[1], vscale=prev[2]))
                arms["core_carried"] = {**score(out, ref, segments), "ms": ms, "carried_from_step": step - 1}
                r, c = msev.rel_cos_lean(out, ours_out)
                arms["core_carried"]["vs_ours"] = {"rel_l2": r, "cos": c,
                                                   "segments": segment_stats(out, ours_out, segments)}
                del out, _km, _vs
            else:
                arms["core_carried"] = {"skipped": "step-1 of this block and branch was not captured"
                                        if prev is None or prev[0] != step - 1 else "unknown"}
            carry[(render, block, branch)] = (step, km_next, vs_next)
            del ours_out

            if args.shipped:
                sh_sinks = ours_mod._sink_blocks(ours_to, s_len, shipped["sink"])
                out, ms = timed(lambda: ours_call(shipped["tau"], shipped["token_aug"], sh_sinks))
                arms["ours_shipped"] = {**score(out, ref, segments), "ms": ms,
                                        "sinks": [list(x) for x in sh_sinks]}
                del out

            row["arms"] = arms
            rec["cells"].append(row)
            carried = arms["core_carried"]
            print(f"  r{render} b{block:>2} s{step:>2} S={s_len}: exact floor ours {arms['control_ours_all_routed']['rel_l2']:.5f} "
                  f"core {arms['control_core_all_routed']['rel_l2']:.5f} | ours {arms['ours']['rel_l2']:.5f} "
                  f"core_fresh {arms['core_fresh']['rel_l2']:.5f} "
                  + (f"core_carried {carried['rel_l2']:.5f}" if "rel_l2" in carried else "core_carried -")
                  + f" | core_fresh vs ours {arms['core_fresh']['vs_ours']['rel_l2']:.5f}"
                  + ("" if row["sinks"]["match"] else "  SINKS DIFFER")
                  + (f"  repro q/k/v identical {all(row['reproduction'][n]['identical'] for n in 'qkv')}"
                     if "reproduction" in row else ""))
            del q, k, v, ref, qkv, d, bhnd
            torch.cuda.empty_cache()

    names = sorted({a for c in rec["cells"] for a in c["arms"] if "rel_l2" in c["arms"][a]})
    agg = {}
    for a in names:
        got = [c["arms"][a] for c in rec["cells"] if "rel_l2" in c["arms"].get(a, {})]
        agg[a] = {"cells": len(got),
                  "rel_l2_mean": sum(x["rel_l2"] for x in got) / len(got),
                  "cos_mean": sum(x["cos"] for x in got) / len(got),
                  "cos_worst_head": min(x["cos_worst"] for x in got),
                  "rel_l2_row_mean": sum(x["rel_l2_row_mean"] for x in got) / len(got)}
        kinds = sorted({k for x in got for k in (x.get("segments") or {})})
        agg[a]["segments_rel_l2_mean"] = {
            k: sum(x["segments"][k]["rel_l2"] for x in got if x.get("segments") and k in x["segments"])
            / max(1, sum(1 for x in got if x.get("segments") and k in x["segments"])) for k in kinds}
    rec["aggregate"] = {**agg, "note": "means over cells weight every (block, step) equally; "
                                       "cos_worst_head is the single worst head anywhere"}
    out_path = Path(args.out) if args.out else HERE.parent / "data" / "results" / f"{time.strftime('%Y-%m-%d')}_sol_impl_capture_grade.json"
    out_path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"\nrecord written to {out_path.relative_to(REPO) if out_path.is_relative_to(REPO) else out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
