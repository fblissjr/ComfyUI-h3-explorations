#!/usr/bin/env python3
"""Calibrate a per-head tau table from tau-sweep records (`sol_tau_sweep.py`).

    <comfy venv python> bench/calibrate_sparse_table.py \\
        --record data/sparse/captures/<sweep A>/sol_sweep_*.jsonl [--record ...] \\
        --stage 10=ref2va_pdd8 --stage 123=ref2va_flashgen_finisher \\
        --target equal_error --dense-blocks 38,39,40,41,42,49 \\
        --out data/results/<date>_sparse_table_<what>.json [--write-tables]

A sweep record holds, per call, head and segment, Sol's squared error against
the fallback at every tau of a grid and the key blocks it routed. A table is
one tau per head for each block of one stage (one sampler node of the swept
graph), so everything is pooled over the stage's steps.

**The budget is the shipped tau's own error, not a free number** (`--base-tau`,
default the swept node's `tau`). Per stage and block, over the video rows:

  equal_error   fewest routed key blocks, with the block's pooled error on
                every scene no higher than at the shipped tau.
  equal_share   lowest worst-scene error, with no more routed key blocks than
                the shipped tau.

Both under a guard: no head's relative error, on any scene, above the worst
head's at the shipped tau in that block, so a table never runs a head worse
than one already running. `--routed-segments` adds a conditioning segment
class whose rows are to run routed (`rows` = per segment on the node): its
error at the shipped tau is then a budget of its own, and its routed blocks
count as cost.

**Held out.** With two scenes or more, each is left out in turn, the table
fitted on the rest and scored on it. A block's margin is its worst held-out
excess, and the table written is fitted on every scene with that margin taken
off the budget (`--margin` sets one number for all blocks instead).

Blocks in `--dense-blocks` run on the dense kernel in the shipped graphs, so
there is no shipped Sol error to hold them to: they get no table row and are
reported apart, with the kernel times the sweep measured on them.

The error is local, per call, against the dense INT8 kernel on the fallback's
trajectory. Local error did not predict the owner's verdicts on 2026-10-02
(`bench/results/2026-10-02_sol_dense_blocks_panel.md`); a table calibrated
here has been measured, not judged, and its provenance says so.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import sparse_table  # noqa: E402

VIDEO = "video"
#: Lagrange multipliers tried per weight vector, as powers of ten.
#: **Reasoned**: cost and error are both normalised to about one per block, so
#: six decades either side of one covers every trade a grid of eight taus has.
LAMBDA_EXPONENTS = np.linspace(-6.0, 6.0, 145)
#: Weight vectors over the error budgets, when there is more than one.
#: **Reasoned**: the corners and centre plus seeded Dirichlet draws; the
#: greedy pass after it closes what a coarse set leaves.
WEIGHT_DRAWS = 48


def segment_class(kind: str) -> str:
    """`sol_attn_h3.segment_class`, with the target video named."""
    if kind in ("video", "text", "audio"):
        return kind
    return "reference"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load(paths) -> tuple[dict, list, dict]:
    """Scenes from sweep records: `{label: {stage: arrays}}`, the tau grid and
    the node settings per stage. A scene is one rendered prompt."""
    scenes, taus, settings = {}, None, {}
    for path in paths:
        renders, configs, cells = {}, {}, {}
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                kind = row["kind"]
                if kind == "header":
                    if taus is not None and row["taus"] != taus:
                        raise SystemExit(f"{path}: tau grid {row['taus']} differs from {taus}")
                    taus = row["taus"]
                elif kind == "render":
                    renders[row["prompt_id"]] = row
                elif kind == "config":
                    configs[row["digest"]] = row["settings"]
                elif kind == "cell":
                    cells.setdefault(row["prompt_id"], []).append(row)
        for prompt_id, rows in cells.items():
            rendered = (renders.get(prompt_id) or {}).get("rendered") or {}
            label = rendered.get("prompt_id") or f"prompt_{str(rendered.get('prompt_sha256'))[:8]}"
            if label in scenes:
                label = f"{label}_{prompt_id[:8]}"
            scenes[label] = _scene(rows, taus, path)
            scenes[label]["rendered"] = {k: rendered.get(k) for k in ("prompt_id", "prompt_sha256", "length", "canvas", "seed")}
            scenes[label]["record"] = str(path)
            for row in rows:
                settings.setdefault(row["executing_node_id"], configs.get(row["config"]))
    if not scenes:
        raise SystemExit("no cells in the records given")
    return scenes, taus, settings


def _scene(rows, taus, path) -> dict:
    stages = {}
    for row in rows:
        stages.setdefault(row["executing_node_id"], []).append(row)
    out = {"stages": {}, "tokens": rows[0]["T"], "segments": rows[0]["segments"]}
    for stage, cells in stages.items():
        blocks = sorted({c["block"] for c in cells})
        steps = sorted({c["sigma"] for c in cells}, reverse=True)
        if len(cells) != len(blocks) * len(steps) or blocks != list(range(len(blocks))):
            raise SystemExit(f"{path}: stage {stage} has {len(cells)} cells for {len(blocks)} blocks "
                             f"and {len(steps)} steps; a sweep must hold each once")
        H, T = cells[0]["H"], len(taus)
        classes = [segment_class(kind) for _a, _b, kind in cells[0]["segments"]]
        names = sorted(set(classes))
        shape = (len(blocks), T, H)
        st = {"blocks": len(blocks), "steps": steps, "heads": H, "ntb": cells[0]["NTB"],
              "num": {n: np.zeros(shape) for n in names}, "routed": {n: np.zeros(shape) for n in names},
              "den": {n: np.zeros((len(blocks), H)) for n in names},
              "query_blocks": {n: 0 for n in names},
              "ms_sol": np.zeros((len(blocks), T)), "ms_fallback": np.zeros(len(blocks)),
              "routed_all_by_cell": [], "ms_by_cell": [], "floor": None}
        for j, cls in enumerate(classes):
            st["query_blocks"][cls] += cells[0]["query_blocks"][j]
        if all(c.get("floor") for c in cells):
            st["floor"] = {"num": {n: np.zeros((len(blocks), H)) for n in names},
                           "den": {n: np.zeros((len(blocks), H)) for n in names}}
        for c in cells:
            if [segment_class(k) for _a, _b, k in c["segments"]] != classes or c["B"] != 1:
                raise SystemExit(f"{path}: stage {stage} block {c['block']}: the layout or the batch changed "
                                 f"within a stage (batch {c['B']})")
            b = c["block"]
            st["ms_fallback"][b] += c["ms"]["fallback"]
            for i, tau in enumerate(taus):
                key = str(tau)
                total = 0.0
                for j, cls in enumerate(classes):
                    st["num"][cls][b, i] += c["num"][key][j]
                    st["routed"][cls][b, i] += c["routed"][key][j]
                    total += float(sum(c["routed"][key][j]))
                st["ms_sol"][b, i] += c["ms"]["sol"][key]
                st["routed_all_by_cell"].append(total)
                st["ms_by_cell"].append(c["ms"]["sol"][key])
            for j, cls in enumerate(classes):
                st["den"][cls][b] += c["den"][j]
                if st["floor"] is not None:
                    st["floor"]["num"][cls][b] += c["floor"]["num"][j]
                    st["floor"]["den"][cls][b] += c["floor"]["den"][j]
        out["stages"][stage] = st
    return out


# ---------------------------------------------------------------------------
# One block: choose a tau per head
# ---------------------------------------------------------------------------

def _weights(n: int) -> np.ndarray:
    if n == 1:
        return np.ones((1, 1))
    rng = np.random.default_rng(0)
    return np.concatenate([np.eye(n), np.full((1, n), 1.0 / n), rng.dirichlet(np.ones(n), WEIGHT_DRAWS)])


def solve(n: np.ndarray, budget: np.ndarray, cost: np.ndarray, allowed: np.ndarray, base: int,
          target: str) -> np.ndarray:
    """A tau index per head. `n` is (constraints, taus, heads) squared error,
    `budget` (constraints,) what each may sum to, `cost` (taus, heads) routed
    key blocks, `allowed` (taus, heads), `base` the shipped tau's index.

    Candidates from a Lagrangian sweep, then a greedy pass of single-head moves.
    The uniform shipped tau is always a candidate."""
    S, T, H = n.shape
    heads = np.arange(H)
    e = n / budget[:, None, None]                                   # (S, T, H), about 1/H each at the base
    c = cost / max(float(cost[base].sum()), 1.0)
    blocked = np.where(allowed, 0.0, np.inf)
    mixed = np.einsum("ws,sth->wth", _weights(S), e)                # (W, T, H)
    lam = 10.0 ** LAMBDA_EXPONENTS
    obj = c[None, None] + lam[None, :, None, None] * mixed[:, None] + blocked[None, None]
    picks = obj.argmin(axis=2).reshape(-1, H)                       # (W * L, H)
    picks = np.concatenate([np.full((1, H), base), picks])
    ratios = e[:, picks, heads].sum(axis=2).T                       # (candidates, S)
    costs = c[picks, heads].sum(axis=1)
    worst = ratios.max(axis=1)
    if target == "equal_error":
        ok = worst <= 1.0 + 1e-12
        if not ok.any():
            raise ValueError("no assignment meets the error budget")
        pick = picks[np.where(ok, costs, np.inf).argmin()].copy()
    else:
        ok = costs <= 1.0 + 1e-12
        pick = picks[np.where(ok, worst, np.inf).argmin()].copy()

    # Greedy: single-head moves that improve the objective and keep the constraint.
    while True:
        cur_e = e[:, pick, heads].sum(axis=1)                       # (S,)
        cur_c = float(c[pick, heads].sum())
        new_e = cur_e[:, None, None] - e[:, pick, heads][:, None, :] + e      # (S, T, H)
        new_c = cur_c - c[pick, heads][None, :] + c                           # (T, H)
        new_worst = new_e.max(axis=0)
        if target == "equal_error":
            gain = np.where(allowed & (new_worst <= 1.0 + 1e-12), cur_c - new_c, -np.inf)
        else:
            gain = np.where(allowed & (new_c <= 1.0 + 1e-12), float(cur_e.max()) - new_worst, -np.inf)
        t, h = np.unravel_index(gain.argmax(), gain.shape)
        if not gain[t, h] > 1e-9:
            return pick
        pick[h] = t


def guard(num: list, den: list, base: int) -> np.ndarray:
    """(taus, heads) allowed: a head's relative error on every scene no higher
    than the worst head's at the shipped tau on that scene."""
    allowed = np.ones(num[0].shape, dtype=bool)
    for n, d in zip(num, den):
        rel = n / np.maximum(d[None], 1e-300)
        allowed &= rel <= rel[base].max() * (1.0 + 1e-9)
    allowed[base] = True
    return allowed


def fit_block(stage_of: list, b: int, base: int, target: str, routed_segments, margin: float):
    """A tau index per head for block `b`, fitted on the scenes in `stage_of`
    (their arrays for one stage)."""
    constraints, budgets = [], []
    for st in stage_of:
        for seg in (VIDEO, *routed_segments):
            constraints.append(st["num"][seg][b])
            budgets.append(st["num"][seg][b, base].sum() * (1.0 - margin))
    cost = sum(st["routed"][seg][b] for st in stage_of for seg in (VIDEO, *routed_segments)) / len(stage_of)
    allowed = guard([st["num"][VIDEO][b] for st in stage_of], [st["den"][VIDEO][b] for st in stage_of], base)
    return solve(np.stack(constraints), np.array(budgets), cost, allowed, base, target)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation of two vectors (ties broken by order; these are floats)."""
    rx, ry = np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def head_error(st: dict, b: int, base: int) -> np.ndarray:
    """Per head, the relative squared error of block `b`'s video rows at the shipped tau."""
    return st["num"][VIDEO][b, base] / np.maximum(st["den"][VIDEO][b], 1e-300)


def score(st: dict, b: int, pick: np.ndarray, base: int, seg: str = VIDEO) -> dict:
    """What an assignment does on one scene's block: error and routed blocks,
    against the shipped tau."""
    heads = np.arange(len(pick))
    num, den, routed = st["num"][seg][b], st["den"][seg][b], st["routed"][seg][b]
    n_t, n_0 = float(num[pick, heads].sum()), float(num[base].sum())
    r_t, r_0 = float(routed[pick, heads].sum()), float(routed[base].sum())
    rel = num[pick, heads] / np.maximum(den, 1e-300)
    return {"err": math.sqrt(n_t / den.sum()), "err_base": math.sqrt(n_0 / den.sum()),
            "err_ratio": math.sqrt(n_t / n_0) if n_0 > 0 else None,
            "routed_ratio": r_t / r_0 if r_0 > 0 else None, "routed": r_t, "routed_base": r_0,
            "worst_head": math.sqrt(float(rel.max())),
            "worst_head_base": math.sqrt(float((num[base] / np.maximum(den, 1e-300)).max()))}


# ---------------------------------------------------------------------------
# The kernel's time as a function of what it routes
# ---------------------------------------------------------------------------

def time_model(scenes: dict) -> dict:
    """ms of one Sol call = a + b * routed key blocks summed over heads and
    query blocks, least squares over every swept call."""
    x = np.array([v for sc in scenes.values() for st in sc["stages"].values() for v in st["routed_all_by_cell"]])
    y = np.array([v for sc in scenes.values() for st in sc["stages"].values() for v in st["ms_by_cell"]])
    A = np.stack([np.ones_like(x), x], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ np.array([a, b])
    return {"ms_intercept": float(a), "ms_per_routed_block": float(b), "calls": int(len(x)),
            "r2": float(1.0 - resid.var() / y.var()), "resid_ms_p95": float(np.quantile(np.abs(resid), 0.95))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_blocks(spec: str) -> set:
    return {int(x) for x in spec.split(",") if x.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="append", required=True, help="a sol_sweep_*.jsonl; repeat per record")
    ap.add_argument("--stage", action="append", required=True, metavar="NODE_ID=NAME",
                    help="a sampler node of the swept graph and the name its table gets")
    ap.add_argument("--target", choices=("equal_error", "equal_share"), default="equal_error")
    ap.add_argument("--base-tau", type=float, default=None, help="default: the swept node's own tau")
    ap.add_argument("--dense-blocks", default="", help="blocks the shipped graph keeps dense: no table row")
    ap.add_argument("--routed-segments", default="", help="conditioning classes whose rows will run routed, "
                                                          "e.g. 'reference'")
    ap.add_argument("--margin", type=float, default=None,
                    help="fraction taken off every squared-error budget; default: each block's worst held-out excess")
    ap.add_argument("--out", required=True, help="the calibration record (JSON)")
    ap.add_argument("--write-tables", action="store_true", help="write sparse_tables/<NAME>_<target>.json")
    args = ap.parse_args()

    scenes, taus, settings = load(args.record)
    labels = sorted(scenes)
    dense = parse_blocks(args.dense_blocks)
    routed_segments = tuple(s.strip() for s in args.routed_segments.split(",") if s.strip())
    model = time_model(scenes)
    record = {"tool": "bench/calibrate_sparse_table.py", "date": datetime.date.today().isoformat(),
              "records": [str(p) for p in args.record], "taus": taus, "target": args.target,
              "dense_blocks": sorted(dense), "routed_segments": list(routed_segments),
              "scenes": {k: {"rendered": scenes[k]["rendered"], "tokens": scenes[k]["tokens"],
                             "segments": scenes[k]["segments"], "record": scenes[k]["record"]} for k in labels},
              "time_model": model, "stages": {}}
    print(f"scenes: {labels}; taus {taus}; time model {model['ms_intercept']:.1f} ms + "
          f"{model['ms_per_routed_block'] * 1e6:.2f} ms per million routed blocks, r2 {model['r2']:.4f}")

    for spec in args.stage:
        stage, _, name = spec.partition("=")
        if any(stage not in scenes[k]["stages"] for k in labels) or not name:
            raise SystemExit(f"--stage {spec}: every scene must hold sampler node {stage}, and the table needs a name")
        sts = [scenes[k]["stages"][stage] for k in labels]
        base_tau = args.base_tau if args.base_tau is not None else float((settings.get(stage) or {}).get("tau", 1.0))
        if base_tau not in taus:
            raise SystemExit(f"the shipped tau {base_tau} is not in the swept grid {taus}")
        base = taus.index(base_tau)
        n_blocks, H = sts[0]["blocks"], sts[0]["heads"]
        table_blocks = [b for b in range(n_blocks) if b not in dense]
        out = {"name": name, "base_tau": base_tau, "steps": sts[0]["steps"], "heads": H, "blocks": {},
               "dense_blocks": {}, "held_out": {}, "summary": {}}

        # Held out: fit on the others, score on the one left out.
        margins = {b: 0.0 for b in table_blocks}
        for i, label in enumerate(labels if len(labels) > 1 else []):
            rest = [st for j, st in enumerate(sts) if j != i]
            rows, fitted, ranks = {}, [], []
            for b in table_blocks:
                pick = fit_block(rest, b, base, args.target, routed_segments, 0.0)
                rows[b] = {seg: score(sts[i], b, pick, base, seg) for seg in (VIDEO, *routed_segments)}
                fitted.append(float(np.mean([score(st, b, pick, base)["routed_ratio"] for st in rest])))
                ranks.append(float(np.mean([spearman(head_error(sts[i], b, base), head_error(st, b, base))
                                            for st in rest])))
                excess = max(rows[b][seg]["err_ratio"] ** 2 - 1.0 for seg in rows[b])
                margins[b] = max(margins[b], excess)
            ratios = np.array([rows[b][VIDEO]["err_ratio"] for b in table_blocks])
            routed = np.array([rows[b][VIDEO]["routed_ratio"] for b in table_blocks])
            guard_broken = [b for b in table_blocks
                            if rows[b][VIDEO]["worst_head"] > rows[b][VIDEO]["worst_head_base"] * (1.0 + 1e-9)]
            fit_mean = float(np.mean(fitted))
            out["held_out"][label] = {
                "err_ratio": {"median": float(np.median(ratios)), "p90": float(np.quantile(ratios, 0.9)),
                              "max": float(ratios.max()), "blocks_over": int((ratios > 1.0).sum())},
                "routed_ratio": {"median": float(np.median(routed)), "mean": float(routed.mean())},
                # The transfer test: the saving in routed blocks on the scenes the table was fitted on, and
                # how much of it is still there on the scene it never saw.
                "routed_ratio_on_fit_scenes": fit_mean,
                "saving_surviving": (1.0 - float(routed.mean())) / (1.0 - fit_mean) if fit_mean < 1.0 else None,
                "guard_broken_blocks": guard_broken,
                "head_error_rank_correlation": {"median": float(np.median(ranks)), "min": float(np.min(ranks)),
                                                "by_block": {str(b): r for b, r in zip(table_blocks, ranks)}},
                "blocks": {str(b): rows[b] for b in table_blocks}}
            print(f"  stage {stage} held out {label}: error ratio median {np.median(ratios):.3f} "
                  f"p90 {np.quantile(ratios, 0.9):.3f} max {ratios.max():.3f} ({int((ratios > 1).sum())} of "
                  f"{len(ratios)} blocks over), routed ratio mean {routed.mean():.3f} against {fit_mean:.3f} "
                  f"where fitted, guard broken on {len(guard_broken)} blocks, head rank correlation median "
                  f"{np.median(ranks):.3f} min {np.min(ranks):.3f}")
        if args.margin is not None:
            margins = {b: args.margin for b in table_blocks}
        if args.target == "equal_share":
            margins = {b: 0.0 for b in table_blocks}        # a margin is a budget's; this target has none

        # The table: fitted on every scene, the margin off the budget.
        table, saved_ms, base_ms = {}, 0.0, 0.0
        cond_blocks = sum(v for k, v in sts[0]["query_blocks"].items() if k != VIDEO)
        for b in table_blocks:
            try:
                pick = fit_block(sts, b, base, args.target, routed_segments, min(margins[b], 0.9))
            except ValueError:
                pick = np.full(H, base)                     # the shipped tau: nothing sparser meets the margin
            table[b] = [taus[i] for i in pick]
            per_scene = {k: {seg: score(st, b, pick, base, seg) for seg in (VIDEO, *routed_segments)}
                         for k, st in zip(labels, sts)}
            entry = {"taus": table[b], "margin": margins[b], "scenes": per_scene,
                     "heads_per_tau": {str(t): int((pick == i).sum()) for i, t in enumerate(taus) if (pick == i).any()}}
            # Predicted time of the shipped call (conditioning rows exact) at the shipped tau and with the table.
            for k, st in zip(labels, sts):
                exact = sum(st["query_blocks"][c] for c in st["query_blocks"] if c != VIDEO and c not in routed_segments)
                fixed = model["ms_intercept"] * len(st["steps"]) + model["ms_per_routed_block"] * (
                    exact * st["ntb"] * H * len(st["steps"]))
                var = lambda idx: model["ms_per_routed_block"] * sum(                      # noqa: E731
                    float(st["routed"][seg][b][idx, np.arange(H)].sum()) for seg in (VIDEO, *routed_segments))
                entry["scenes"][k]["ms_base"] = fixed + var(np.full(H, base))
                entry["scenes"][k]["ms_table"] = fixed + var(pick)
            base_ms += float(np.mean([entry["scenes"][k]["ms_base"] for k in labels]))
            saved_ms += float(np.mean([entry["scenes"][k]["ms_base"] - entry["scenes"][k]["ms_table"] for k in labels]))
            if sts[0]["floor"] is not None or any(st["floor"] is not None for st in sts):
                st = next(st for st in sts if st["floor"] is not None)
                fl = st["floor"]["num"][VIDEO][b] / np.maximum(st["floor"]["den"][VIDEO][b], 1e-300)
                rel = st["num"][VIDEO][b][pick, np.arange(H)] / np.maximum(st["den"][VIDEO][b], 1e-300)
                ratio = np.sqrt(rel / np.maximum(fl, 1e-300))
                entry["floor"] = {"pooled": math.sqrt(float(st["floor"]["num"][VIDEO][b].sum() / st["floor"]["den"][VIDEO][b].sum())),
                                  "table_over_floor_per_head": {"min": float(ratio.min()), "median": float(np.median(ratio)),
                                                                "max": float(ratio.max())}}
            out["blocks"][str(b)] = entry

        for b in sorted(dense):
            if b >= n_blocks:
                continue
            out["dense_blocks"][str(b)] = {
                "err_by_tau": {str(t): float(np.mean([math.sqrt(st["num"][VIDEO][b, i].sum() / st["den"][VIDEO][b].sum())
                                                      for st in sts])) for i, t in enumerate(taus)},
                "ms_sol_rows_routed_by_tau": {str(t): float(np.mean([st["ms_sol"][b, i] / len(st["steps"]) for st in sts]))
                                              for i, t in enumerate(taus)},
                "ms_fallback": float(np.mean([st["ms_fallback"][b] / len(st["steps"]) for st in sts]))}

        errs = np.array([[out["blocks"][str(b)]["scenes"][k][VIDEO]["err_ratio"] for k in labels] for b in table_blocks])
        routed = np.array([[out["blocks"][str(b)]["scenes"][k][VIDEO]["routed_ratio"] for k in labels] for b in table_blocks])
        out["summary"] = {"table_blocks": len(table_blocks), "cond_query_blocks": cond_blocks,
                          "err_ratio_worst_scene": {"median": float(np.median(errs.max(axis=1))), "max": float(errs.max())},
                          "video_routed_ratio_mean": float(routed.mean()),
                          "predicted_ms_per_render_base": base_ms, "predicted_ms_per_render_saved": saved_ms,
                          "margin": {"median": float(np.median(list(margins.values()))), "max": float(max(margins.values()))}}
        print(f"stage {stage} ({name}), {args.target}: video routed blocks x{routed.mean():.3f}, worst-scene error "
              f"x{errs.max(axis=1).mean():.3f} (max {errs.max():.3f}); predicted Sol time {base_ms / 1e3:.1f} s per render "
              f"at the shipped tau, {saved_ms / 1e3:.1f} s less with the table")
        record["stages"][stage] = out

        if args.write_tables:
            doc = {"schema": sparse_table.SCHEMA,
                   "provenance": {"calibrated_on": args.out, "tool": record["tool"], "date": record["date"],
                                  "target": args.target, "base_tau": base_tau, "stage_node": stage,
                                  "scenes": labels, "routed_segments": list(routed_segments),
                                  "dense_blocks_left_out": sorted(dense),
                                  "judged": "no: calibrated on local error against the dense INT8 kernel; "
                                            "nothing rendered with it has been looked at"},
                   "heads": H, "blocks": {str(b): table[b] for b in table_blocks}}
            sparse_table.parse(doc, name)
            path = sparse_table.TABLE_DIR / f"{name}_{args.target}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(doc, indent=1) + "\n")
            record["stages"][stage]["table_file"] = str(path.relative_to(REPO))
            print(f"  wrote {path.relative_to(REPO)}")

    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
