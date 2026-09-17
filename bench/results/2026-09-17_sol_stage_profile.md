# Where a Sol call spends its time, and whether the replay matches a render

Date: 2026-09-17. Model: MiniMax H3, int8 convrot checkpoint, base 16-step
t2v, 345 frames at 1344x768 (packed length 104361, 56 heads). RTX 4090, the
card alone for the replay. comfy-kitchen 0.2.34+sol.b532e28 (the owner's fork,
`h3-build`). Pack at ab1283e plus the timer hook this record introduces.

Files:

- `2026-09-17_sol_stages.json`: replay, `bench/profile_sol_stages.py`, 15 cells
  (blocks 0, 24, 32, 40, 49 at steps 4, 8, 15), all heads, four arms.
- `2026-09-17_sol_live_call_times.jsonl`: one real render of
  `workflows/h3_text_to_video_api.json` (market scene, seed 730451892) with
  `H3_SOL_TIME` armed (`sol_call_timer.py`): one row per attention call.
- `2026-09-17_default_chain_walltime_baseline.jsonl` and `..._timed.jsonl`:
  the same graph unarmed and armed.

## 1. The instrument adds up

Coverage (summed kernel time over the same call timed with no profiler) is
0.98 to 1.01 on every cell and arm. The stages account for the whole call;
there is no host gap to attribute.

## 2. The split

Shipped recipe (tau 1.0, pooled tail, `qk_balance` on, the node's own sinks):

| stage | share of the call |
|---|---|
| exact | 0.92 to 0.94 |
| preprocess | 0.05 to 0.06 |
| vtranspose | 0.01 |
| route (incl. the pooled tail) | under 0.005 |

The same at every captured block and step. Exact-stage time is linear in
routed density: the cells run from 0.217 to 0.262 routed and from 193 to
236 ms of exact time, one line through the origin to within two percent.
Routing itself is free; what it routes is the cost.

What each option adds per call, median over the 15 cells, and to which stage:

| option | added ms | stage |
|---|---|---|
| `qk_balance` | 2.6 | preprocess |
| `rotate` (with balance) | 31.6 | preprocess, which goes from about 9 to about 41 ms |
| token routing, budget 64 | 23.2 | its own stage |

## 3. The replay is the render

| block | replay, shipped arm, ms | live, median of the 12 Sol steps, ms | live / replay |
|---|---|---|---|
| 0 | 220.2 | 218.4 | 0.992 |
| 24 | 219.3 | 214.3 | 0.977 |
| 32 | 247.7 | 247.0 | 0.997 |
| 40 | 228.4 | 220.4 | 0.965 |
| 49 | 210.0 | 207.8 | 0.989 |

So a Sol call costs the same inside a render as alone, and the stage split
above can be read as the render's. Arming cost nothing measurable: 508 s armed
against 501 and 509 s unarmed (total, submit to finish).

## 4. What the render pays for attention

From the live rows, one pass per step (50 calls per step, batch 1):

| | steps | calls | per call, median ms | seconds |
|---|---|---|---|---|
| dense fallback (kitchen int8), before `start_percent` | 4 | 200 | 719.6 | 143.4 |
| Sol | 12 | 599 of 600 (the last call of a process stays unread) | 219.5 | 133.4 |

Sampler time for the render was 469 s, so attention is 277 s of it, and
**the four dense steps cost more than the twelve Sol steps**. A dense step is
about 36 s of attention, a Sol step about 11 s.

By block, live: the slowest Sol calls are blocks 31 to 36 (242 to 252 ms), the
fastest are 49, 45, 13 and 10 (208 to 213 ms); the spread is routed density.

## 5. What this says to do

- `start_percent` is the largest lever on this graph by a wide margin: each
  dense step turned sparse is about 25 s, and there are four. Its quality cost
  is unmeasured; renders at 0.2, 0.1 and 0.0 are
  `2026-09-17_sol_start_percent_arms.jsonl`, scored by eye.
- Inside Sol, only the exact stage is worth kernel time. A fifth off it (what
  Comfy-Org/comfy-kitchen#184 reports for its paired-query-block walk on HIP)
  would be about 25 s of this render. Routing and the tail are not worth
  touching.
- `rotate`'s cost is all preprocess and is about 19 s of this render
  (31.6 ms over 600 calls). A parallel rotation can recover most of that;
  it matters only if `rotate` becomes a default.
- Token routing on every block would be about 14 s.

## 6. Staging-bound or compute-bound, answered without counters

GPU performance counters are admin-only on this machine, so `ncu` could not
run. Substitute: three compile-time probe builds of the exact kernel (stage
only, compute only, neither), timed against the kernel as written;
`2026-09-17_sol_exact_stage_vs_compute.json`.

| variant | exact stage, share of the kernel as written |
|---|---|
| stage only | about 0.70 |
| compute only | about 0.91 |
| neither | about 0.03 |

Compute is the critical path and the one-block-ahead async staging hides most
of the copy time behind it. Removing staging entirely would save about a
tenth, which bounds what comfy-kitchen#184's shared staging can give here;
making compute free would save at most three tenths. No single change to this
stage is worth a kernel week. The levers are outside it: `start_percent` and
routed density.
