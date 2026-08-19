#!/usr/bin/env python3
"""Is per-head granularity worth anything, priced against just moving the global tau?

`bench/join_density_error.py` established that a head's routed density does not
predict its Sol error, which is the precondition for per-head tuning being
worth something: if error were inferable from spend, there would be nothing to
tune toward. This asks the next question, which is the one that decides it --
**does any per-head arm beat the global tau on the same trade?**

Every arm here spends the same currency (routed density, the fraction of key
blocks kept exact) to buy the same thing (less error against dense attention).
So they are directly comparable on one ratio: density spent per fraction of
error removed, relative to the shipped operating point. Lower is better, and an
arm that does not beat the global tau is not worth its plumbing however
interesting the structure that motivated it.

## The three arms and why each is the honest form of its idea

**Global tau.** The control, and the one every per-head proposal has to beat. It
is already a per-head threshold in a real sense: the routing threshold is
derived from per-head key-centroid statistics, so tau scales a quantity that
already differs per head.

**Per-head dense escape.** Take the k heads with the most error at the operating
point and run them exact. This is the honest form of "the bad heads need
something other than a threshold nudge", and it is priced exactly rather than
extrapolated: the dense limit is measured, not modelled -- `--control`'s
`tau=-1e9` arm gives the per-head error floor this substitutes in.

**Per-head tau is deliberately NOT an arm here, and that is a result.** Pricing
it needs each head's error as a function of tau, and two operating points give a
linear estimate over the interval between them. Every attempt to equalise error
that way put 45 to 52 of 56 heads outside the measured interval, because tau is
a weak lever on a head's error next to the spread between heads -- so the
estimate would be extrapolation reported as measurement. The tau-leverage table
this prints is that finding rather than a diagnostic.

## What this cannot tell you

**Mean per-head relative error is a proxy for output quality and not the thing
itself.** It weights a head with a small output norm equally with a large one,
where the aggregate does not. Two arms with equal mean per-head error are not
thereby known to look the same, and CLAUDE.md's rule stands: a rendered clip
cannot A/B a numerical change on any sampler, so nothing here can be settled by
looking at output.

**Every input record's caveats survive.** The density side is the eager routing
rule rather than the CUDA kernel's INT8 arithmetic; the error side is measured
against this repo's eager Sol, gated against the vendored oracle; and the dense
floor is measured at one cell and assumed to hold at the others, which is the
one substitution here that is not a per-cell measurement.

## Running it

    python bench/price_head_arms.py \\
        --density bench/results/2026-08-19_routing_density_per_head.json \\
        --error bench/results/2026-08-19_sol_error_per_head.json \\
        --error-alt bench/results/2026-08-19_sol_error_per_head_tau1.0.json \\
        --control bench/results/2026-08-19_sol_error_control.json \\
        --out bench/results/2026-08-19_head_granularity_arms.json

No GPU, no capture, no server: it reads four records.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path


def load(path, want_capture=None):
    rec = json.loads(Path(path).read_text())
    if want_capture and rec.get("capture") != want_capture:
        raise SystemExit(
            f"{Path(path).name} is of capture {rec.get('capture')}, not "
            f"{want_capture}. Arms priced across two workloads are not arms.")
    return rec


def per_head_error(rec):
    """{(block, step, head): sparsity error}, refusing a measured head prefix."""
    out = {}
    for r in rec["rows"]:
        if r["heads_measured"] != r["heads_total"]:
            raise SystemExit(
                f"{r['heads_measured']} of {r['heads_total']} heads at b{r['block']} "
                f"s{r['step']}: that slice is a PREFIX, not a sample, so its head "
                f"indices do not name the heads the density record does. "
                f"Re-run analyze_sol_error.py with --heads 0.")
        for head, value in enumerate(r["per_head_sparsity"]):
            out[(r["block"], r["step"], head)] = value
    return out


def mean_over_cells(cells, heads, fn):
    return st.mean(st.mean(fn(b, s, h) for h in heads) for b, s in cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--density", required=True)
    ap.add_argument("--error", required=True, help="error at the operating tau")
    ap.add_argument("--error-alt", required=True,
                    help="error at a second tau, for the leverage table")
    ap.add_argument("--control", required=True,
                    help="a --control record; its dense-limit arm is the escape floor")
    ap.add_argument("--dense-k", default="3,5,8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dens = load(args.density)
    capture = dens["capture"]
    err_rec = load(args.error, capture)
    alt_rec = load(args.error_alt, capture)
    ctl_rec = load(args.control, capture)

    tau, tau_alt = float(err_rec["tau"]), float(alt_rec["tau"])
    if tau == tau_alt:
        raise SystemExit(
            f"both error records are at tau {tau}; the leverage table needs two "
            f"operating points and the arms need to know which is shipped.")

    err = per_head_error(err_rec)
    alt = per_head_error(alt_rec)
    dm = {(r["tau"], r["block"], r["step"], r["head"]): r["kernel_density_pct"]
          for r in dens["rows"]}
    for t in (tau, tau_alt):
        if not any(k[0] == t for k in dm):
            raise SystemExit(
                f"the density record has no rows at tau {t}, so that arm cannot "
                f"be priced. Re-run sweep_routing_density.py with it in --taus.")

    dense_arm = next((c for c in ctl_rec["controls"] if "dense_limit" in c["arm"]), None)
    if dense_arm is None:
        raise SystemExit(
            f"{Path(args.control).name} carries no dense_limit arm, so the escape "
            f"floor would have to be assumed rather than measured.")
    floor = st.median(dense_arm["per_head_sparsity"])

    cells = sorted({(b, s) for b, s, _ in err})
    heads = sorted({h for _, _, h in err})
    n = len(heads)

    # Leverage: what a tau move does to one head, against what separates heads.
    print(f"TAU LEVERAGE  (tau {tau_alt} -> {tau})\n")
    print(f"  {'cell':<10}{'per-head error response':>25}{'between-head spread':>22}")
    leverage = []
    for b, s in cells:
        resp = [err[(b, s, h)] / alt[(b, s, h)] for h in heads if alt[(b, s, h)] > 0]
        vals = [err[(b, s, h)] for h in heads]
        spread = max(vals) / min(vals) if min(vals) > 0 else None
        row = {"block": b, "step": s,
               "error_response_median_x": round(st.median(resp), 4),
               "error_response_min_x": round(min(resp), 4),
               "error_response_max_x": round(max(resp), 4),
               "between_head_spread_x": round(spread, 2) if spread else None}
        leverage.append(row)
        print(f"  b{b:<3}s{s:<5}"
              f"{'x%.2f (%.2f-%.2f)' % (st.median(resp), min(resp), max(resp)):>25}"
              f"{'%.1fx' % spread if spread else 'n/a':>22}")
    med_resp = st.median(r["error_response_median_x"] for r in leverage)
    med_spread = st.median(r["between_head_spread_x"] for r in leverage)
    print(f"\n  A tau move of this size changes a head's error by a median x{med_resp:.2f}.")
    print(f"  Heads differ from each other by a median {med_spread:.1f}x at one point.")
    print(f"  That is why per-head tau is not an arm below: bringing heads together")
    print(f"  needs tau moves far outside anything measured here.\n")

    # Arms, all priced against the shipped operating point.
    arms = []

    def add(label, dfn, efn, detail=None):
        arms.append({"arm": label,
                     "mean_kernel_density_pct": round(mean_over_cells(cells, heads, dfn), 4),
                     "mean_per_head_sparsity_error": round(mean_over_cells(cells, heads, efn), 6),
                     **(detail or {})})

    add(f"global tau {tau} (shipped)",
        lambda b, s, h: dm[(tau, b, s, h)], lambda b, s, h: err[(b, s, h)])
    add(f"global tau {tau_alt}",
        lambda b, s, h: dm[(tau_alt, b, s, h)], lambda b, s, h: alt[(b, s, h)])
    for k in (int(x) for x in args.dense_k.split(",")):
        top = {(b, s): set(sorted(heads, key=lambda h: -err[(b, s, h)])[:k])
               for b, s in cells}
        add(f"tau {tau} + top-{k} heads dense",
            lambda b, s, h, top=top: 100.0 if h in top[(b, s)] else dm[(tau, b, s, h)],
            lambda b, s, h, top=top: floor if h in top[(b, s)] else err[(b, s, h)],
            {"dense_heads_per_cell": k})

    base = arms[0]
    print("ARMS  (density spent per fraction of error removed; lower is better)\n")
    print(f"  {'arm':<32}{'mean density':>14}{'mean per-head err':>19}{'ratio':>10}")
    for a in arms:
        if a is base:
            a["density_per_unit_error_cut"] = None
            ratio = "baseline"
        else:
            cut = (base["mean_per_head_sparsity_error"]
                   - a["mean_per_head_sparsity_error"]) / base["mean_per_head_sparsity_error"]
            spend = (a["mean_kernel_density_pct"]
                     - base["mean_kernel_density_pct"]) / base["mean_kernel_density_pct"]
            a["density_per_unit_error_cut"] = round(spend / cut, 4) if cut > 0 else None
            ratio = f"{a['density_per_unit_error_cut']:.2f}x" if cut > 0 else "n/a"
        print(f"  {a['arm']:<32}{a['mean_kernel_density_pct']:13.2f}%"
              f"{a['mean_per_head_sparsity_error']:19.4f}{ratio:>10}")

    # Concentration: how much of the error the worst heads actually hold.
    conc = []
    for k in (1, 3, 5, 8, 12):
        share = []
        for b, s in cells:
            v = sorted((err[(b, s, h)] for h in heads), reverse=True)
            share.append(100 * sum(v[:k]) / sum(v))
        conc.append({"k": k, "share_pct": round(st.mean(share), 2),
                     "uniform_pct": round(100 * k / n, 2)})
    print("\nERROR CONCENTRATION  (mean over cells)\n")
    for c in conc:
        print(f"  top {c['k']:<3}of {n}: {c['share_pct']:5.1f}% of summed per-head error"
              f"   (uniform would be {c['uniform_pct']:.1f}%)")

    record = {
        "measured": "2026-08-19",
        "produced_by": "bench/price_head_arms.py",
        "what": "per-head granularity arms priced against the global tau, one currency",
        "capture": capture,
        "operating_tau": tau,
        "comparison_tau": tau_alt,
        "dense_limit_floor_used": floor,
        "dense_limit_floor_source": "the --control record's dense_limit arm, measured "
                                    "at one cell and applied at all of them",
        "heads": n,
        "cells": [{"block": b, "step": s} for b, s in cells],
        "inputs": [Path(p).name for p in (args.density, args.error,
                                          args.error_alt, args.control)],
        "argv": sys.argv[1:],
        "tau_leverage": leverage,
        "arms": arms,
        "error_concentration": conc,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
        print(f"\nwrote {args.out}")
    else:
        json.dump(record, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
