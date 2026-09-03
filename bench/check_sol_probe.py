#!/usr/bin/env python3
"""Controls for the Sol-versus-fallback probe (`sol_block_probe.py`), and the
reader that turns its record into a per-block table.

Three modes, because the spec names three kinds of evidence:

  --controls           synthetic fixtures on the GPU, small: armed and unarmed
                       outputs bitwise equal; unarmed writes nothing and makes
                       no fallback call; trajectory=sage returns the fallback's
                       output bitwise; a stock-attention reference is reported
                       as not sage; identical and perturbed inputs give the
                       metric its expected values; skips are recorded with a
                       reason; segment and completeness validators go red on
                       shuffled boundaries, a duplicate cell and a missing one.
  --record PATH        read a probe jsonl: per-block table (whole-call rel L2
                       mean, p50, p90, p99, max over its cells; per-segment),
                       completeness against the schedule the rows carry, and
                       every invariant above. Nonzero exit on a violation.
  --replay-capture DIR the metric against the retained capture: for each
                       qkv_*.pt, Sol all-routed and the fp32 reference exactly
                       as bench/measure_sol_exact_variants.py computed them,
                       through `sol_block_probe.metrics`; the whole-call rel L2
                       and per-head cosines must reproduce the record named
                       by --against. GPU, tens of minutes at production length.

Green here means the instrument's arithmetic and bookkeeping hold on
fixtures; what the shipped call's numbers ARE is the record's business.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Record validation (no GPU)
# ---------------------------------------------------------------------------

def load_record(path: Path) -> tuple[dict | None, list[dict]]:
    header, rows = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "header":
            header = row
        else:
            rows.append(row)
    return header, rows


def validate(rows: list[dict], n_blocks: int | None = None) -> list[str]:
    """Every invariant the spec names; returns the violations."""
    bad = []
    cells = [r for r in rows if r.get("kind") == "cell"]
    skips = [r for r in rows if r.get("kind") == "skip"]
    if not cells:
        bad.append("no cell rows")
        return bad
    # one prompt id per render, cells joined to it
    by_prompt = defaultdict(list)
    for r in cells:
        by_prompt[r.get("prompt_id")].append(r)
    if None in by_prompt:
        bad.append(f"{len(by_prompt[None])} cell(s) carry no prompt id")
    for pid, cs in by_prompt.items():
        # segments contiguous over [0, T)
        for r in cs:
            T = r.get("T"); segs = r.get("segments")
            if segs:
                spans = sorted((int(a), int(b)) for a, b, _k in segs)
                if spans[0][0] != 0 or spans[-1][1] != T or any(spans[i][1] != spans[i + 1][0] for i in range(len(spans) - 1)):
                    bad.append(f"prompt {pid} block {r.get('block')} sigma {r.get('sigma')}: segments {spans} do not tile [0, {T})")
            m = r.get("metrics")
            if r.get("compare_status") == "compared" and m:
                w = m["whole"]
                for k in ("numerator", "denominator", "diff_rms", "ref_rms"):
                    v = w.get(k)
                    if v is None or not math.isfinite(v):
                        bad.append(f"prompt {pid} block {r.get('block')}: whole.{k} is {v}")
                if w.get("rel_l2") is not None and not math.isfinite(w["rel_l2"]):
                    bad.append(f"prompt {pid} block {r.get('block')}: rel_l2 not finite")
            elif r.get("compare_status") != "compared":
                bad.append(f"prompt {pid} block {r.get('block')} sigma {r.get('sigma')}: compare_status {r.get('compare_status')}: {r.get('compare_reason')}")
        # completeness: every (schedule index, block) seen by Sol exactly once
        seen = defaultdict(int)
        for r in cs:
            key = (r.get("schedule", {}).get("schedule_index"), r.get("block"))
            seen[key] += 1
        dups = {k: n for k, n in seen.items() if n > 1}
        if dups:
            bad.append(f"prompt {pid}: duplicate cells {sorted(dups.items())[:5]}")
        steps = sorted({k[0] for k in seen if k[0] is not None})
        blocks_seen = sorted({k[1] for k in seen if k[1] is not None})
        nb = n_blocks or (max(blocks_seen) + 1 if blocks_seen else 0)
        expected = {(s, b) for s in steps for b in range(nb)}
        missing = sorted(expected - set(seen))
        # a missing cell is legitimate only if a skip row explains it
        explained = {(r.get("schedule", {}).get("schedule_index"), r.get("block")) for r in skips if r.get("prompt_id") == pid}
        unexplained = [k for k in missing if k not in explained]
        if unexplained:
            bad.append(f"prompt {pid}: {len(unexplained)} (step, block) cell(s) neither compared nor skipped, e.g. {unexplained[:5]}")
        if any(k[0] is None for k in seen):
            bad.append(f"prompt {pid}: cells with no schedule index (sigma not on the schedule)")
    return bad


def table(rows: list[dict]) -> str:
    cells = [r for r in rows if r.get("kind") == "cell" and r.get("compare_status") == "compared"]
    by_block = defaultdict(list)
    for r in cells:
        by_block[r["block"]].append(r)
    out = ["  block  cells   rel_l2 mean     p50     p90     p99     max   worst-head cos   segments (rel_l2 mean)"]
    import statistics
    def pct(xs, p):
        xs = sorted(xs); i = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
        return xs[i]
    ranked = []
    for b, cs in by_block.items():
        vals = [c["metrics"]["whole"]["rel_l2"] for c in cs if c["metrics"]["whole"]["rel_l2"] is not None]
        if not vals:
            continue
        worst_head = min((h["cos"] for c in cs for h in c["metrics"]["per_head"] if h["cos"] is not None), default=None)
        segs = defaultdict(list)
        for c in cs:
            for sgm in c["metrics"]["per_segment"]:
                if sgm["rel_l2"] is not None:
                    segs[sgm["kind"]].append(sgm["rel_l2"])
        ranked.append((statistics.mean(vals), b, len(cs), vals, worst_head, segs))
    for mean, b, n, vals, wh, segs in sorted(ranked, reverse=True):
        seg_txt = " ".join(f"{k}={statistics.mean(v):.4f}" for k, v in sorted(segs.items()))
        out.append(f"  {b:5d}  {n:5d}   {mean:.5f}  {pct(vals, .5):.5f} {pct(vals, .9):.5f} {pct(vals, .99):.5f} {max(vals):.5f}   "
                   f"{'' if wh is None else f'{wh:.4f}':>8}   {seg_txt}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Synthetic controls (GPU, small)
# ---------------------------------------------------------------------------

def _load_pack():
    """The pack as a package, with ComfyUI importable, the way
    `check_sol_node_equivalence.py` does it: the node modules use relative
    imports and `comfy_api`, so a bare `import sol_attn_h3` cannot work."""
    import importlib.util
    comfy = REPO.parent.parent
    if str(comfy) not in sys.path:
        sys.path.insert(0, str(comfy))
    def load(name, path, package_dir=None):
        spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=[str(package_dir)] if package_dir else None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    if "h3x" not in sys.modules:
        load("h3x", REPO / "__init__.py", package_dir=REPO)
    import h3x.sol_block_probe as probe           # noqa: E402
    import h3x.attention as _attn                 # noqa: E402
    import h3x.sol_attn_h3 as solh3               # noqa: E402
    return probe, _attn, solh3


def controls() -> int:
    import torch
    probe, _attn, solh3 = _load_pack()

    def exact(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kw):
        # a stock-attention stand-in with optimized_attention's signature, BHND in, BHND out
        out = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float()).to(q.dtype)
        return out if skip_output_reshape else out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    fn, kw = _attn.build_kernel("auto")
    sage_override = _attn.make_sage_override(fn, kw, previous=None)
    settings = {"node": "MiniMaxH3SolAttn", "tau": 1.0, "n_blocks": 4, "dense_blocks": [2]}
    torch.manual_seed(0)
    B, H, T, D = 1, 8, 4096, 128
    q = (torch.randn(B, H, T, D, device="cuda") * 0.5).bfloat16()
    k = (torch.randn(B, H, T, D, device="cuda") * 0.5).bfloat16()
    v = torch.randn(B, H, T, D, device="cuda").bfloat16()
    k[:, :, 0] *= 8   # a sink-like key so routing is non-trivial
    sample_sigmas = [1.0, 0.8, 0.5, 0.2, 0.0]
    def opts(block, sigma=0.5):
        return {"sigmas": torch.tensor([sigma]), "sample_sigmas": sample_sigmas, "sol_block": block,
                "h3_segments": [(0, 128, "text"), (128, 256, "audio"), (256, T, "video")],
                "sol_h3_video_span": (256, T), "sol_h3_audio_span": (128, 256)}
    results = []
    def case(name, ok, detail=""):
        results.append(ok); print(f"  {'ok  ' if ok else 'FAIL'}  {name:44} {detail}")

    with tempfile.TemporaryDirectory(prefix="h3_probe_controls_") as tmp:
        sol = solh3.make_override(tau=1.0, min_tokens=1024, previous=sage_override, settings=settings,
                                  dense_blocks=frozenset({2}), sink_conditioning="exact_kv")
        # 1. unarmed: no record, no fallback call, output A
        probe.arm("")
        import sageattention as sa
        c0 = probe._count_total(sa.get_dispatch_counts())
        out_unarmed = sol(exact, q, k, v, H, skip_reshape=True, skip_output_reshape=True, transformer_options=opts(0))
        c1 = probe._count_total(sa.get_dispatch_counts())
        case("unarmed makes no fallback call", (c1 - c0) == 0, f"sage dispatch delta {c1 - c0}")
        case("unarmed writes no record", probe.path() is None)
        # 2. armed trajectory=sol: bitwise equal output, one fallback call, a cell row
        probe.arm(f"dir={tmp},trajectory=sol,capture=controls")
        c0 = probe._count_total(sa.get_dispatch_counts())
        out_armed = sol(exact, q, k, v, H, skip_reshape=True, skip_output_reshape=True, transformer_options=opts(0))
        c1 = probe._count_total(sa.get_dispatch_counts())
        case("armed output bitwise equals unarmed", torch.equal(out_armed, out_unarmed))
        case("armed made exactly one fallback call", (c1 - c0) == 1, f"delta {c1 - c0}")
        header, rows = load_record(Path(probe.path()))
        cell = [r for r in rows if r["kind"] == "cell"]
        case("one cell row, compared, returned sol", len(cell) == 1 and cell[0]["compare_status"] == "compared" and cell[0]["returned_backend"] == "sol",
             f"{[c['compare_status'] for c in cell]}")
        case("counterfactual telemetry says sage ran", cell and cell[0]["counterfactual"]["sage_dispatch_delta"] == 1,
             f"{cell[0]['counterfactual'] if cell else None}")
        m = cell[0]["metrics"] if cell else None
        case("metrics carry segments and per-head rows", bool(m) and m["segments_recorded"] and len(m["per_head"]) == H and
             [s["kind"] for s in m["per_segment"]] == ["text", "audio", "video"])
        # 3. a dense_blocks call is a skip row with its reason
        sol(exact, q, k, v, H, skip_reshape=True, skip_output_reshape=True, transformer_options=opts(2))
        _, rows = load_record(Path(probe.path()))
        sk = [r for r in rows if r["kind"] == "skip"]
        case("dense block recorded as a skip with reason", len(sk) == 1 and sk[0]["route"] == "dense_block", f"{[s.get('reason') for s in sk]}")
        # 4. trajectory=sage returns the fallback's output bitwise
        probe.arm(f"dir={tmp},trajectory=sage,capture=controls")
        ref_direct = sage_override(exact, q, k, v, H, skip_reshape=True, skip_output_reshape=True, transformer_options=opts(0))
        out_sage = sol(exact, q, k, v, H, skip_reshape=True, skip_output_reshape=True, transformer_options=opts(0))
        _, rows = load_record(Path(probe.path()))
        case("trajectory=sage returns the fallback bitwise", torch.equal(out_sage, ref_direct) and rows[-1]["returned_backend"] == "sage")
        # 5. a stock-attention reference (no sage chained) is reported as not sage
        probe.arm(f"dir={tmp},trajectory=sol")
        sol_stock = solh3.make_override(tau=1.0, min_tokens=1024, previous=None, settings=settings, sink_conditioning="exact_kv")
        sol_stock(exact, q, k, v, H, skip_reshape=True, skip_output_reshape=True, transformer_options=opts(0))
        _, rows = load_record(Path(probe.path()))
        case("stock reference reported as reference_not_sage", rows[-1]["compare_status"] == "reference_not_sage", rows[-1]["compare_status"])
        # 6. the metric on identical and perturbed inputs
        x = torch.randn(1, 4, 512, 128, device="cuda").bfloat16()
        ident = probe.metrics(x, x, heads=4, skip_output_reshape=True, segments=[(0, 256, "a"), (256, 512, "b")])
        case("identical inputs: rel_l2 0, cos 1", ident["whole"]["rel_l2"] == 0.0 and abs(ident["whole"]["cos"] - 1.0) < 1e-6)
        pert = probe.metrics((x.float() * 1.01).bfloat16(), x, heads=4, skip_output_reshape=True, segments=[(0, 256, "a"), (256, 512, "b")])
        case("1% scaled input: rel_l2 about 0.01", abs(pert["whole"]["rel_l2"] - 0.01) < 0.003, f"{pert['whole']['rel_l2']:.4f}")
        case("per-head rows summarised with count", all(h["rows"]["count"] == 512 for h in pert["per_head"]))
        z = x.clone(); z[:, :, :10] = 0
        zr = probe.metrics(x, z, heads=4, skip_output_reshape=True)
        case("zero-reference rows excluded and counted", zr["whole"]["zero_reference_rows"] == 40 and all(h["rows"]["count"] == 502 for h in zr["per_head"]))
        # 7. the record validator goes red on shuffled segments, a duplicate, a missing cell
        probe.arm(f"dir={tmp},trajectory=sol,capture=validator")
        for blk in (0, 1, 3):
            for sig in (0.8, 0.5):
                sol(exact, q, k, v, H, skip_reshape=True, skip_output_reshape=True, transformer_options=opts(blk, sig))
            sol(exact, q, k, v, H, skip_reshape=True, skip_output_reshape=True, transformer_options=opts(2, 0.5)) if blk == 0 else None
        _, rows = load_record(Path(probe.path()))
        good = validate(rows, n_blocks=4)
        # block 2 at sigma 0.8 was never called: neither compared nor skipped -> must be flagged
        case("validator flags the never-called cell", any("neither compared nor skipped" in b for b in good), f"{good[:1]}")
        rows2 = [dict(r) for r in rows]
        for r in rows2:
            if r["kind"] == "cell" and r["block"] == 1:
                r["segments"] = [[128, 256, "audio"], [0, 128, "text"], [300, T, "video"]]
        case("validator flags non-tiling segments", any("do not tile" in b for b in validate(rows2, n_blocks=4)))
        rows3 = rows + [dict(r) for r in rows if r["kind"] == "cell"][:1]
        case("validator flags a duplicate cell", any("duplicate" in b for b in validate(rows3, n_blocks=4)))
        rows4 = [dict(r) for r in rows]
        for r in rows4:
            if r["kind"] == "cell":
                r["prompt_id"] = None
        case("validator flags cells with no prompt id", any("no prompt id" in b for b in validate(rows4, n_blocks=4)))
        probe.arm("")
    n_bad = results.count(False)
    print(f"\n{'all ok -- the probe behaves as specified on fixtures' if not n_bad else f'{n_bad} control(s) failed'}")
    return 1 if n_bad else 0


# ---------------------------------------------------------------------------
# Replay against the retained capture (GPU, heavy)
# ---------------------------------------------------------------------------

def replay(capture: Path, against: Path, chunk: int) -> int:
    import glob
    import torch
    import comfy_kitchen as ck
    probe, _a, _s = _load_pack()
    sys.path.insert(0, str(REPO / "bench"))
    from measure_sol_exact_variants import dense_fp32_chunked
    rec = json.loads(against.read_text(encoding="utf-8"))
    per_file = {r["file"]: r for r in rec["arms"]["per_file"]}
    bad = 0
    for f in sorted(glob.glob(str(capture / "qkv_*.pt"))):
        name = os.path.basename(f)
        if name not in per_file:
            print(f"  skip  {name}: not in the record"); continue
        d = torch.load(f, map_location="cuda", mmap=False)
        q, k, v = (d[n].permute(0, 2, 1, 3).contiguous().to(torch.bfloat16) for n in ("q", "k", "v"))
        ref = dense_fp32_chunked(q, k, v, chunk)              # BTHD fp32
        out = ck.sol_attn(q, k, v, tau=-1e9)                  # BTHD bf16, every block routed
        m = probe.metrics(out.transpose(1, 2), ref.transpose(1, 2), heads=q.shape[2], skip_output_reshape=True)
        want = per_file[name]["exact_all_routed"]
        rel_ok = abs(m["whole"]["rel_l2"] - want["rel_l2"]) <= 1e-6 * max(1.0, want["rel_l2"]) + 1e-9
        cos_ok = abs(m["whole"]["cos"] - want["cos"]) <= 1e-6
        bad += (not (rel_ok and cos_ok))
        print(f"  {'ok  ' if rel_ok and cos_ok else 'FAIL'}  {name}: rel_l2 {m['whole']['rel_l2']:.6f} vs record {want['rel_l2']:.6f}; cos {m['whole']['cos']:.6f} vs {want['cos']:.6f}")
        del d, q, k, v, ref, out
        torch.cuda.empty_cache()
    print("\n" + ("the metric reproduces the record on every cell" if not bad else f"{bad} cell(s) differ from the record"))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--record", type=Path)
    ap.add_argument("--n-blocks", type=int, default=None, help="expected block count for completeness (default: max seen + 1)")
    ap.add_argument("--replay-capture", type=Path)
    ap.add_argument("--against", type=Path, help="the measure_sol_exact_variants record to reproduce")
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()
    rc = 0
    if args.controls:
        rc |= controls()
    if args.record:
        header, rows = load_record(args.record)
        print(f"record {args.record.name}: trajectory {header.get('trajectory') if header else '?'}, "
              f"{sum(r.get('kind') == 'cell' for r in rows)} cells, {sum(r.get('kind') == 'skip' for r in rows)} skips")
        print(table(rows))
        bad = validate(rows, args.n_blocks)
        for b in bad:
            print(f"  FAIL  {b}")
        print("  ok    every invariant holds" if not bad else f"\n{len(bad)} violation(s)")
        rc |= 1 if bad else 0
    if args.replay_capture:
        if not args.against:
            print("--replay-capture needs --against RECORD"); return 2
        rc |= replay(args.replay_capture, args.against, args.chunk)
    if not (args.controls or args.record or args.replay_capture):
        ap.print_help(); return 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
