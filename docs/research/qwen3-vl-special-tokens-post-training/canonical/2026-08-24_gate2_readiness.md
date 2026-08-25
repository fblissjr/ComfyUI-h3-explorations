# Gate 2 readiness and resume point

**Status:** Authoritative for the requirements it states; the pending Gate 2A
measurement was accepted on 2026-08-25 and its result, together with the
Gate 2B arrangement, is owned by
[`2026-08-25_gate2_arrangement.md`](2026-08-25_gate2_arrangement.md)
**Recorded:** 2026-08-24
**Corrected producer commit:** `745d916`
**Boundary:** no AWQ recipe, candidate checkpoint, deployment change, or launch
authorization

## What is closed

Gate 1 proved the native-H3 presentation-to-`llm-compressor` seam on real
single-image, multi-image, keyframe, mixed keyframe-plus-reference, and
reference-video rows. Gate 1B accepted `comfy_exact` as the calibration-only
precision policy. The evidence and limitations remain owned by
[`2026-08-24_gate1_seam_acceptance.md`](2026-08-24_gate1_seam_acceptance.md)
and
[`2026-08-24_transformers_comfy_parity.md`](2026-08-24_transformers_comfy_parity.md).

**MEASURED, bounded diagnostic.** The pinned stack can convert an Accelerate
load through the supported `llm-compressor`/`compressed_tensors` offload bridge
and execute the no-modifier sequential path with all 64 decoder layers present.
The converted model had no remaining Accelerate hooks or `hf_device_map`.
This closes the substrate question only: it does not establish the resource
budget for an AWQ run.

The raw `device_map="auto"` failure does not show that sequential compression
and offload are incompatible. That unconverted path left a wrapped
`forward` that the sequential machinery could not call in the required form.
The supported bridge replaces those hooks; subsequent work must use and audit
that converted path rather than generalize from the raw-load failure.

## What is not accepted

The resource table committed with the first Gate 2A diagnostic is superseded as
a budgeting source. Its instrumentation could onload parameters while
describing model placement, reset CUDA peaks inside the interval it intended to
measure, follow staging symlinks while counting disk use, and frame failures by
population-total tokens even though the sequential path executes one row at a
time. Those results remain useful as a correction record, not as a calibration
population limit.

No automatic SDPA backend has been identified from an availability query.
Availability and selection are separate questions, and both depend on the
actual tensor shape and causal mode. The earlier 512-token synthetic query also
omitted the cuDNN candidate and cannot explain a failure on a substantially
larger real row. No memory mechanism or KV-expansion remedy is accepted from
that query.

Gate 2B has not started. No modifier has been instantiated, no quantized
weights have been produced, and no absolute calibration population has been
accepted.

The committed Gate 2A producer is
[`pilot_sequential_feasibility.py`](../../../../bench/pilot_sequential_feasibility.py).
The separate dispatch instrument is
[`probe_sdpa_backend_selection.py`](../../../../bench/probe_sdpa_backend_selection.py).
The corrected final result files are intentionally absent at this stopping
point; tomorrow's rerun must create them from commit `745d916` or record and
review any subsequent producer change.

## Gate 2A acceptance record required

Promote the no-modifier floor only from one coherent rerun produced by one
committed harness version. The record must include:

- all 64 decoder layers loaded through the supported converted offload bridge,
  with remaining Accelerate hooks, `hf_device_map`, and conversion failures
  reported explicitly;
- placement of parameters and buffers read from the offload mechanism itself,
  without traversing model parameters in a way that onloads them;
- a cumulative CUDA peak spanning cache construction and all sequential
  forwards, plus per-call entry residency and transient growth where those
  measurements are valid;
- current process RSS, process high-water RSS, and system memory available,
  kept as distinct quantities;
- physical staging bytes, files, and symlinks, with unique symlink targets
  deduplicated rather than silently followed and double-counted;
- total population tokens, longest individual row, and each row's visual-block
  grids reported separately;
- the original allocator message and whether an OOM occurred during
  intermediate-cache construction or a subgraph forward;
- backend availability queried at every actual text sequence length and unique
  vision-block shape, including Flash, memory-efficient, and cuDNN candidates;
- actual automatic-backend selection, if claimed, established by a separate
  profiler or forced-backend probe rather than inferred from availability; and
- cleanup after completion, deliberate abort, and OOM, with every partial
  output marked non-launchable.

Nominal attention-logit tensor sizes may be recorded as mechanism clues. They
must be labelled as nominal footprints, not predicted allocator requests, and
interpreted only beside the allocator message, failure stage, and same-shape
backend evidence.

Gate 2A remains a modifier-free floor. GPU feasibility is governed by the
largest row and visual block that must execute, while population size also
changes the host-side intermediate cache. Neither a completed population-token
sum nor an OOM boundary between two prefixes is, by itself, a launch budget.

## Gate 2B entry contract

After the corrected Gate 2A record is accepted, run the smallest disposable
experiment that instantiates the real AWQ modifier and exercises its observer
and sequential state. It must:

1. load a fresh model for every modifier arm because AWQ mutates modules and
   attaches state;
2. use the supported converted offload bridge with an explicit host-memory
   reserve chosen from the corrected Gate 2A record rather than inheriting the
   bridge default;
3. apply the accepted `comfy_exact` precision policy and effective-input mask
   transform;
4. construct the recipe in the pinned environment, with AWQ observer offload
   explicitly placed on CPU and the decoder-only quantization boundary checked
   before execution;
5. measure modifier state, cache/replay, CUDA, host memory, staging, time, and
   cleanup against the no-modifier floor;
6. include an intentional abort and a mutation capable of proving that the
   modifier path was actually entered; and
7. have no candidate output path, serializer, symlink action, or publishing
   step.

Gate 2B may determine an absolute population only if the measured
modifier-bearing arms cover the accepted workload shapes and leave a declared
operating reserve. Otherwise it narrows the next measurement; it does not
authorize a guessed population.

## Resume order

1. Commit and review the corrected, single-version Gate 2A artifacts and their
   producer.
2. Review the separate backend-selection probe only for the shapes it actually
   measured; do not make it a prerequisite unless its result changes the
   resource interpretation or Gate 2B design.
3. Update this record with accepted Gate 2A measurements and provenance.
4. Implement and run the non-exporting Gate 2B modifier pilot.
5. Freeze the component-safe calibration/holdout split and launch package only
   after Gate 2B establishes a defensible budget.

The deployed W4 checkpoint, its processor snapshot and symlink, the BF16 source
tree, and the special-token training boundary remain unchanged.
