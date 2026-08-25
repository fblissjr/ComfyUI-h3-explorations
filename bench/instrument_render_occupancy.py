"""Phase-resolved GPU occupancy of one render, bracketed per node.

Answers one question about a graph: during its sampler node, is the card
busy or waiting on the host? A kernel-launch-bound phase shows the GPU idle
between short kernels (low `utilization.gpu`, power well under the limit);
a compute- or power-bound phase shows the opposite. That is the measurement
that decides whether CUDA-graph replay could buy anything on a graph, before
anyone builds it: graph replay removes launch gaps and nothing else.

Method, the same one `bench/results/2026-08-18_phase0_instrument.json` used
by hand with `nvidia-smi dmon` at 1 s: sample `nvidia-smi --query-gpu` at
100 ms in a subprocess (dmon's floor of 1 s is too coarse for a render that
lasts 13 s), follow the server's websocket to record every node's start and
end wall time, then bracket the samples by node. `utilization.gpu` is the
fraction of the sample period during which any kernel was executing, so
launch gaps of tens of microseconds between kernels of the same order show
up as a percentage well below 100; it says nothing finer than that, and this
script claims nothing finer.

Verdict rule for the sampler window, normative (change it here, not in
prose): LAUNCH-BOUND SIGNAL when mean utilization.gpu < 80 and mean power <
90% of the board limit; COMPUTE/POWER-BOUND when mean utilization.gpu >= 95
or mean power >= 95% of the limit; MIXED otherwise. The verdict line prints
the metric values beside the thresholds.

Usage (needs a live server and an idle card; refuses if the card is busy):

    python bench/instrument_render_occupancy.py --graph workflows/image/h3_image_edit_api.json \\
        --label image_edit --seed 7 --out bench/results/2026-08-26_image_edit_occupancy.json

Controls that need no card:

    python bench/instrument_render_occupancy.py --self-test
    python bench/instrument_render_occupancy.py --replay-dmon bench/results/2026-08-18_phase0_dmon.log

`--self-test` pushes a synthetic launch-bound and a synthetic compute-bound
sample set through the verdict and fails unless each reads as itself.
`--replay-dmon` summarises an independent record (the dmon log behind the
2026-08-18 instrument, sampling proxied as rows with framebuffer >= 20000
MiB) beside what that record's JSON states, so the summariser is checked
against numbers it did not compute.

The seed is applied to every RandomNoise node: a byte-identical resubmission
returns the server's cached sampler output as a 0.0 s "render"
(`bench/run_graph_arms.py` records the instance), and a cached run has no
sampler window to measure.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import hashlib
import json
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "results"))   # make_attention_defaults_json.substrate

QUERY_FIELDS = ("timestamp", "power.draw", "power.limit", "utilization.gpu",
                "utilization.memory", "clocks.sm", "clocks.max.sm",
                "memory.used")
SAMPLE_MS = 100
SAMPLER_CLASSES = ("SamplerCustomAdvanced", "KSampler", "KSamplerAdvanced",
                   "SamplerCustom")

# Verdict thresholds. Normative: the rule lives here.
LAUNCH_UTIL_BELOW = 80.0
LAUNCH_POWER_BELOW_PCT = 90.0
BOUND_UTIL_AT_LEAST = 95.0
BOUND_POWER_AT_LEAST_PCT = 95.0

BUSY_UTIL_PCT = 5          # refuse to measure beside another job
BUSY_MEMORY_MIB = 2048


# --------------------------------------------------------------------------
# summarising
# --------------------------------------------------------------------------

def _pct(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize_window(samples):
    """samples: list of dicts with power_w, limit_w, util_pct, mem_pct,
    pclk_mhz, pclk_max_mhz, fb_mib. Returns the aggregate for one window."""
    if not samples:
        return {"n": 0}
    util = [s["util_pct"] for s in samples]
    power = [s["power_w"] for s in samples]
    mem = [s["mem_pct"] for s in samples]
    pclk = [s["pclk_mhz"] for s in samples if s.get("pclk_mhz") is not None]
    limit = next((s["limit_w"] for s in samples if s.get("limit_w")), None)
    pclk_max = next((s["pclk_max_mhz"] for s in samples
                     if s.get("pclk_max_mhz")), None)
    out = {
        "n": len(samples),
        "util_pct": {"mean": round(statistics.fmean(util), 1),
                     "p50": _pct(util, 0.5), "min": min(util)},
        "power_w": {"mean": round(statistics.fmean(power), 1),
                    "p50": _pct(power, 0.5), "limit": limit},
        "mem_interface_pct": {"mean": round(statistics.fmean(mem), 1),
                              "p50": _pct(mem, 0.5), "p95": _pct(mem, 0.95)},
        "fb_mib_max": max(s["fb_mib"] for s in samples),
    }
    if pclk:
        out["pclk_mhz"] = {"p50": _pct(pclk, 0.5), "max_rated": pclk_max}
    if limit:
        out["power_w"]["mean_pct_of_limit"] = round(
            100.0 * out["power_w"]["mean"] / limit, 1)
    return out


def verdict(summary):
    """The rule, stated with its evidence. Returns (label, evidence)."""
    if summary.get("n", 0) == 0:
        return "NO SAMPLES", "no telemetry rows fell inside the window"
    util = summary["util_pct"]["mean"]
    ppct = summary["power_w"].get("mean_pct_of_limit")
    ev = (f"mean utilization.gpu {util}% (launch signal < {LAUNCH_UTIL_BELOW}, "
          f"bound >= {BOUND_UTIL_AT_LEAST}); mean power "
          f"{summary['power_w']['mean']} W"
          + (f" = {ppct}% of limit (launch signal < {LAUNCH_POWER_BELOW_PCT}, "
             f"bound >= {BOUND_POWER_AT_LEAST_PCT})" if ppct is not None
             else " (limit unknown, power not consulted)"))
    if util >= BOUND_UTIL_AT_LEAST or (ppct is not None
                                       and ppct >= BOUND_POWER_AT_LEAST_PCT):
        return "COMPUTE/POWER-BOUND", ev
    if util < LAUNCH_UTIL_BELOW and (ppct is None
                                     or ppct < LAUNCH_POWER_BELOW_PCT):
        return "LAUNCH-BOUND SIGNAL", ev
    return "MIXED", ev


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------

def _parse_query_row(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(QUERY_FIELDS):
        return None
    try:
        ts = _dt.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
    except ValueError:
        return None

    def num(x):
        try:
            return float(x)
        except ValueError:
            return None

    return {
        "t": ts.timestamp(),
        "power_w": num(parts[1]), "limit_w": num(parts[2]),
        "util_pct": num(parts[3]), "mem_pct": num(parts[4]),
        "pclk_mhz": num(parts[5]), "pclk_max_mhz": num(parts[6]),
        "fb_mib": num(parts[7]),
    }


def read_query_log(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        r = _parse_query_row(line)
        if r and None not in (r["power_w"], r["util_pct"], r["mem_pct"],
                              r["fb_mib"]):
            rows.append(r)
    return rows


def read_dmon_log(path):
    """`nvidia-smi dmon -s pucmt` rows -> the same sample dicts. No limit,
    no rated clock: the dmon format carries neither."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 15:
            continue
        try:
            rows.append({
                "t": _dt.datetime.strptime(p[0] + p[1], "%Y%m%d%H:%M:%S").timestamp(),
                "power_w": float(p[3]), "limit_w": None,
                "util_pct": float(p[6]), "mem_pct": float(p[7]),
                "pclk_mhz": float(p[13]), "pclk_max_mhz": None,
                "fb_mib": float(p[14]),
            })
        except ValueError:
            continue
    return rows


def card_is_busy():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()
    util, mem = (float(x) for x in out[0].split(","))
    return (util > BUSY_UTIL_PCT or mem > BUSY_MEMORY_MIB), util, mem


def start_sampler(log_path):
    f = open(log_path, "w")
    proc = subprocess.Popen(
        ["nvidia-smi", f"--query-gpu={','.join(QUERY_FIELDS)}",
         "--format=csv,noheader,nounits", f"-lms", str(SAMPLE_MS)],
        stdout=f, stderr=subprocess.DEVNULL)
    return proc, f


# --------------------------------------------------------------------------
# the render, with node spans
# --------------------------------------------------------------------------

async def render_with_spans(host, graph, timeout_s):
    """Submit and follow the websocket. Returns (spans, error, prompt_id);
    spans is a list of (node_id, t_start, t_end) in wall time."""
    import aiohttp
    from bench_e2e_h3 import http_post

    client_id = uuid.uuid4().hex
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(f"ws://{host}/ws?clientId={client_id}",
                                   heartbeat=30) as ws:
            resp = http_post(f"http://{host}/prompt",
                             {"prompt": graph, "client_id": client_id})
            prompt_id = resp["prompt_id"]
            spans, current, t_node = [], None, None
            deadline = time.monotonic() + timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return spans, f"timed out after {timeout_s:.0f}s", prompt_id
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
                except asyncio.TimeoutError:
                    return spans, f"timed out after {timeout_s:.0f}s", prompt_id
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                mtype, d = data.get("type"), data.get("data", {})
                if d.get("prompt_id") not in (None, prompt_id):
                    continue
                if mtype == "executing":
                    now = time.time()
                    if current is not None:
                        spans.append((current, t_node, now))
                    node = d.get("node")
                    if node is None:
                        return spans, None, prompt_id
                    current, t_node = node, now
                elif mtype == "execution_error":
                    return spans, d.get("exception_message", "execution error"), prompt_id
                elif mtype == "execution_interrupted":
                    return spans, "interrupted", prompt_id


def bracket(rows, spans):
    per_node = {}
    for node, t0, t1 in spans:
        inside = [r for r in rows if t0 <= r["t"] < t1]
        entry = per_node.setdefault(node, {"seconds": 0.0, "samples": []})
        entry["seconds"] += t1 - t0
        entry["samples"].extend(inside)
    return per_node


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def self_test():
    def synth(util, power, limit=450.0, n=50):
        return [{"power_w": power, "limit_w": limit, "util_pct": util,
                 "mem_pct": 30.0, "pclk_mhz": 2700.0, "pclk_max_mhz": 3105.0,
                 "fb_mib": 20000.0} for _ in range(n)]

    cases = [
        ("launch-bound synthetic", synth(45.0, 220.0), "LAUNCH-BOUND SIGNAL"),
        ("compute-bound synthetic", synth(100.0, 446.0), "COMPUTE/POWER-BOUND"),
        ("power-pegged, util 90", synth(90.0, 440.0), "COMPUTE/POWER-BOUND"),
        ("util 85, power 60%", synth(85.0, 270.0), "MIXED"),
        ("empty window", [], "NO SAMPLES"),
    ]
    ok = True
    for name, samples, want in cases:
        got, ev = verdict(summarize_window(samples))
        flag = "ok  " if got == want else "FAIL"
        ok &= got == want
        print(f"  {flag} {name}: {got}  [{ev}]")
    print("self-test", "passed" if ok else "FAILED")
    return 0 if ok else 1


def replay_dmon(path, fb_floor=20000.0):
    rows = read_dmon_log(path)
    sampling = [r for r in rows if r["fb_mib"] >= fb_floor]
    s = summarize_window(sampling)
    label, ev = verdict(s)
    print(f"{path}: {len(rows)} rows, {len(sampling)} with fb >= {fb_floor:.0f} MiB")
    print(json.dumps(s, indent=2))
    print("verdict:", label)
    print("evidence:", ev)
    sibling = Path(path).with_name(Path(path).name.replace("_dmon.log", "_instrument.json"))
    if sibling.exists():
        rec = json.loads(sibling.read_text())["phases"]["sampling"]
        print("recorded in", sibling.name, "->",
              {k: rec[k] for k in ("power_w_mean", "sm_pct", "pclk_mhz_p50")
               if k in rec})
    return 0


def live(args):
    busy, util, mem = card_is_busy()
    if busy:
        raise SystemExit(f"card busy (util {util:.0f}%, {mem:.0f} MiB used): "
                         "a profile taken beside another job is a wrong one")
    graph_path = Path(args.graph)
    raw = graph_path.read_bytes()
    graph = json.loads(raw)
    try:
        rec_path = str(graph_path.resolve().relative_to(REPO))
    except ValueError:
        rec_path = graph_path.name
    classes = {nid: n["class_type"] for nid, n in graph.items()}
    for node in graph.values():
        w = node.get("inputs", {})
        if node["class_type"] == "RandomNoise":
            w["noise_seed"] = args.seed
        if isinstance(w.get("filename_prefix"), str):
            w["filename_prefix"] += f"_occupancy_{args.label}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_name(out.stem + "_query.log")
    proc, fh = start_sampler(log_path)
    time.sleep(1.0)                                  # idle baseline rows
    try:
        spans, err, prompt_id = asyncio.run(
            render_with_spans(args.host, graph, args.timeout))
        time.sleep(1.0)
    finally:
        proc.terminate()
        proc.wait()
        fh.close()

    rows = read_query_log(log_path)
    per_node = bracket(rows, spans)
    phases = {}
    for nid, entry in per_node.items():
        phases[nid] = {"class_type": classes.get(nid, "?"),
                       "seconds": round(entry["seconds"], 2),
                       **summarize_window(entry["samples"])}
    sampler_ids = [nid for nid in per_node
                   if classes.get(nid) in SAMPLER_CLASSES]
    sampler_summary = summarize_window(
        [s for nid in sampler_ids for s in per_node[nid]["samples"]])
    label, ev = verdict(sampler_summary)

    from make_attention_defaults_json import substrate
    record = {
        "measured": _dt.date.today().isoformat(),
        "produced_by": (f"bench/instrument_render_occupancy.py, nvidia-smi "
                        f"--query-gpu at {SAMPLE_MS} ms (raw log beside this "
                        f"file: {log_path.name})"),
        "what": ("per-node GPU occupancy of one render; the sampler verdict "
                 "is the launch-bound test for CUDA-graph replay"),
        "graph": {"path": rec_path, "sha256_16": hashlib.sha256(raw).hexdigest()[:16],
                  "seed": args.seed, "label": args.label},
        "render": {"prompt_id": prompt_id, "error": err,
                   "total_seconds": round(sum(e["seconds"] for e in per_node.values()), 2)},
        "sampler": {"node_ids": sampler_ids, **sampler_summary,
                    "verdict": label, "evidence": ev},
        "phases": phases,
        "substrate": substrate(),
        "rule": {"launch_util_below": LAUNCH_UTIL_BELOW,
                 "launch_power_below_pct": LAUNCH_POWER_BELOW_PCT,
                 "bound_util_at_least": BOUND_UTIL_AT_LEAST,
                 "bound_power_at_least_pct": BOUND_POWER_AT_LEAST_PCT},
    }
    out.write_text(json.dumps(record, indent=2) + "\n")

    print(f"graph {rec_path} seed {args.seed}: "
          f"{record['render']['total_seconds']} s"
          + (f"  ERROR {err}" if err else ""))
    for nid, ph in phases.items():
        if ph["n"] == 0:
            continue
        print(f"  {nid:>4} {ph['class_type']:<32} {ph['seconds']:7.2f} s  "
              f"util {ph['util_pct']['mean']:5.1f}%  power "
              f"{ph['power_w']['mean']:6.1f} W  mem {ph['mem_interface_pct']['mean']:5.1f}%")
    print(f"sampler verdict: {label}")
    print(f"evidence: {ev}")
    print("wrote", out)
    return 1 if err else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", help="API-format graph to render")
    ap.add_argument("--label", default="run", help="tag for outputs and record")
    ap.add_argument("--seed", type=int, default=1,
                    help="applied to every RandomNoise node (see docstring)")
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", help="JSON record path (bench/results/...)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--replay-dmon", metavar="LOG",
                    help="summarise an nvidia-smi dmon log instead of rendering")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.replay_dmon:
        return replay_dmon(args.replay_dmon)
    if not (args.graph and args.out):
        ap.error("--graph and --out are required for a live run")
    return live(args)


if __name__ == "__main__":
    sys.exit(main())
