# Instruments for Sol: what each measures, how to arm it, and the traps

Detail page of [`sol_attn.md`](sol_attn.md). Every environment switch below is read when the server process starts,
so arm it on a fresh start (`docs/working/comfy_notes.md` has the recipe), and check `/proc/<pid>/environ` before
touching a server someone else armed. The node refuses to arm the observer, the probe and the sweep together.

| instrument | arm with | records | what it cannot say |
|---|---|---|---|
| route observer | `H3_SOL_OBSERVE="dir=/path[,raw=0]"` | one JSONL row per override call (every route, not only Sol), plus a uint16 sidecar of the kernel's routed-block counts (183 KiB a call at the canonical geometry) | which key blocks were chosen (`blk_idx` is not recorded) or what the route cost in error. Timing from an armed render is not quotable: each Sol call synchronizes once |
| call timer | `H3_SOL_TIME=/path/calls.jsonl` | device time of every attention call the override handles, route and block included, from CUDA event pairs read once complete (no added sync). Its `dense` rows price the steps outside the window | anything not on the override's path |
| stage profile | `bench/profile_sol_stages.py` on a capture | device time split by kernel into preprocess, vtranspose, route, exact and the token stage, with a `coverage` figure against the unprofiled call | a live render; the call timer is its check |
| tau sweep | `H3_SOL_SWEEP="dir=/path[,taus=...][,floor=1]"` | every attention call runs the fallback once and Sol at each tau of a grid on the same q, k, v; the model gets the fallback's output. One render gives every head, block and step at every tau | it is not a Sol render (the stamp says so, `provenance.py::_returns_fallback`) |
| error decomposition | `bench/analyze_sol_error.py` on a capture | sparsity and INT8 terms against fp32 dense ([`sol_error.md`](sol_error.md)) | anything perceptual; run `--heads 0` for all 56 heads or every row is a first-n-heads figure |
| captures | `H3_CAPTURE` before the server starts (`h3_capture.py`) | real q/k/v at a chosen block and step, about 4.5 GB a cell at 345 frames | they are not permanent; a capture deleted from disk cannot be re-graded |

The schemas are in the module docstrings (`sol_observe.py`, `sol_call_timer.py`, `sol_tau_sweep.py`). The graders:
`bench/check_sol_observe.py` grades the recorder; `bench/grade_sol_record.py` grades a live record structurally
(every forward carries blocks 0-49 once, routes match the window and the dense list, counts only on `sol` rows, CRCs
verify, the prompt id joins `/history`); `bench/sol_observe_report.py` reads one. The kernel's count semantics are
pinned upstream in `tests/test_sol_attn.py` on the branch, as equalities at both tau extremes.

## Reading a route record

- **Identity is read at call time**, never at patch time, because ComfyUI caches the patched model across prompts.
  `prompt_id` comes from `comfy_execution.utils.get_executing_context()` and is the key `/history` uses. The block
  label is set by a pre-hook and cleared by a paired post-hook, so a row without one is `scope: unknown`, not "token
  refiner".
- **Three densities, and the weighting is in the name.** `kernel_density` is `cnt / NTB` with the forced sink and
  diagonal pairs included: the cost. `ordering_effect_density` is `sum(cnt - forced) / sum(NTB - forced)` over rows
  outside `sink_q`, pair-weighted: the number `bench/analyze_routing.py` computes. `routed_density` is the
  distribution of the per-query-block fraction, query-weighted, so its mean differs from the ratio wherever `forced`
  varies (a five-block case reads 0.5667 against 0.5833). Segment figures carry both.
- **Calls the override never sees are recorded from the composition gate** as `route: composed_patch`, with the
  verdict leading the reason (`outside_range: ...` or `ineligible: ...`).
- **The producer asserts its own shape** (floor, ceiling, `sink_q` rows at NTB); a failure aborts the render rather
  than thinning the file.
- **A `render` row per prompt** names the graph it ran (hashed as `provenance.py` hashes it, plus the shipped file it
  is byte-identical to), a summary (PDD LoRA and steps, UNET, sampler, scheduler, canvas, length) and
  `process_render_index`: 0 is the first prompt after a restart. Routed counts are a function of the inputs, so cold
  and warm renders of one graph at one seed must agree bitwise; that is a control.

The first live record (2026-09-01, the canonical PDD graph as shipped then) graded clean and corrected two
expectations before anyone quoted them: the shipped t2v graphs hand the sampler already-projected text, so there are
no sampler-time refiner rows; and the chain assert's probe ran on a fresh thread without the executing context until
`assert_chain.py` copied it in.

## The tau table and per-segment rows

Two optional inputs, both off by default, both served by the kitchen build's `sol_attn(tau_map=...)`
(`bench/results/2026-10-03_kitchen_tau_map.md`):

- **`tau_table`** names a file in `sparse_tables/`: a tau for every head of the blocks it lists. `sparse_table.py`
  owns the format and refuses a table it does not recognise or one calibrated for another model. No table ships.
  `bench/run_tau_sweep.py` queues the scenes and `bench/calibrate_sparse_table.py` fits a table; what was measured
  before the sweep was trusted, and what has not been run, is `bench/results/2026-10-03_tau_sweep.md`.
- **`rows`**, `per segment`, sets text, reference and target-audio query rows exact or routed apart
  ([`sol_sink.md`](sol_sink.md)).

With neither chosen no `tau_map` keyword is passed, so the kernel call is the one the node made before;
`bench/check_sol_node_equivalence.py` holds that by reading the keywords that reach the kernel. The stamp records the
table's name and the hash of its values, and the row setting (`provenance.py::SOL_CLOSURE_KEYS`). **Not done:** the
route observer still computes densities from `sink_q`, so rows made exact through the map read as routed rows; the
settings record says which policy ran. Nothing has been judged with either input.

## Reproducing

```bash
# Sol against the dense kernel, above the token floor
python bench/bench_e2e_h3.py --arms sage,sage+sol --runs 2 --length 362

# start_percent
python bench/bench_e2e_h3.py --length 362 --runs 2 \
    --arms shipped,shipped+start0.0,shipped+start0.1,shipped+start0.3,shipped+start0.4
```

`--sol-backend` defaults to `cuda`; `triton` refuses at argparse since the pack was deleted. Named arms written in
the Triton vocabulary (`int8_qk`/`int8_pv`) refuse under `cuda` rather than silently becoming a different arm. A
render at 362 frames is about 12 minutes, so a 5-point sweep at 2 runs is about 2 hours.

## Measurement traps hit while producing these pages

Each produced a confident, wrong number first.

| trap | what it did |
|---|---|
| ComfyUI caches node outputs | re-submitting an identical graph executed nothing; the first e2e bench reused one seed and reported a 405x speedup for a render that never ran. `/history` is the observable |
| run 1 of a Triton arm paid autotune | a warm arm against a cold one made Morton look 3 points better |
| reasoning from wall clock is not profiling | per-step arithmetic suggested routing overhead ate the benefit; the profile put routing at 2.1% |
| comparing finished renders numerically measures chaos | at 16-20 steps of a flow-matching ODE any perturbation diverges the trajectory; grade on captured activations instead |
| synthetic inputs cannot answer distribution questions | a `torch.randn` sweep reported `smooth_k` as having no effect, which it would have reported either way |
| grading a kernel against the wrong tail mode | about 1.2e-3 of apparent accuracy, more than the gap between the backends (`../method/checks.md`) |
| an arm below the token floor | a null that reads as a finding |
| one arm of a control pair quoted alone | a two-arm result read as the isolated number; opposite signs across a control pair mean no effect, for time as for memory |
| a timing harness polling `/history` | arm times quantized to the poll interval; read the server's own execution span |
