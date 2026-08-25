# Gate 2A accepted, and the arrangement Gate 2B runs on

**Status:** Authoritative execution boundary
**Recorded:** 2026-08-25
**Authority note:** Codex accepted Gate 2A on 2026-08-25 and then went
offline; later that day the owner confirmed Codex is not returning and named
the encoder Claude technical point. Decisions below marked **ACTING-POINT
DECISION** were made under that authority, on the measurements cited, and are
reversible by the owner; none of them launches anything.
**Boundary:** no AWQ recipe, candidate checkpoint, deployment change, or
launch authorization.

## Gate 2A: accepted

**MEASURED**, one committed harness version, reports under `bench/results/`
dated 2026-08-25 (`gate2a_abort_control`, `gate2a_primary_escalation`,
`gate2a_stress_2048`, `sdpa_backend_selection_primary`,
`sdpa_backend_selection_stress`):

- All 64 decoder layers load through the supported converted offload bridge
  with zero Accelerate hooks and no `hf_device_map`. The bridge's disk tier is
  symlinks into the source shards; zero bytes are staged.
- Under grouped-query attention at FP32, decoder layer 0 on the longest row
  dispatches `aten::_scaled_dot_product_attention_math` and materialises the
  `[64, L, L]` logit tensor. The 8,981-token stress row's allocator request
  is that tensor exactly (19.23 GiB). Flash, memory-efficient and cuDNN all
  report no kernel for that shape; the profiler names the op and forced math
  reproduces `auto` bit for bit.
- The full sequential arrangement completed a 5,857-token row and failed the
  6,189-token video row; the direct kernel completes 6,189 and fails 8,981
  and 10,358. The threshold is arrangement-specific and one video fixture
  failing does not exclude the family.
- The mask omission survives the real dataloader, cache and all 65 traced
  subgraphs. Cleanup after completion, deliberate abort and OOM is clean.
- FP32-resident storage peaks at 122.6 GiB host RSS on load and leaves
  5 GiB available after three rows. It cannot host a modifier.

**Corrections consumed**, from Codex's review: population size is one budget
axis among several (row envelope, visual-block envelope, population/cache,
runtime, host reserve), not the wrong axis; all 64 layers stay the artifact
target; the interpolated 300x300-to-2048 stress row is evaluation-only, which
is a decision about manufactured pixels and not about genuine high-resolution
references or the 2048 processor ceiling.

## The storage axis: accepted

**MEASURED.** `comfy_exact_bf16_store`
([`h3_calibration_precision.py`](../../../../bench/h3_calibration_precision.py))
stores and offloads the weights at BF16 and computes every parameterised op
with a transient FP32 copy of its weight, at the functional layer, gated on the
parameterised modules. On the four released-weight Gate 1B fixtures it is
bit-identical to FP32-stored `comfy_exact` at the raw layer-49 state
(relative L2 0.0 on all, text and vision rows; identical hidden-state hashes;
`bench/results/2026-08-25_storage_axis_layer49_*.json`). Through the bridge
it loads in 0.2 s with 62 GiB on CPU, nothing on disk, and `MemAvailable`
holding at 118 to 120 GiB across the run
(`2026-08-25_gate2a_primary_bf16_store.json`). The guard is
[`check_calibration_precision_policy.py`](../../../../bench/check_calibration_precision_policy.py);
its leak controls refuse a BF16 patch embed at entry and a BF16 activation at
the first linear.

**ACTING-POINT DECISION.** `comfy_exact_bf16_store` is the Gate 2B and
candidate-calibration storage arrangement. It is bit-identical to the accepted
Gate 1B policy, so it inherits that acceptance rather than needing its own.

**SOURCE, for the precision record.** cuDNN TF32 is on by default in both
stacks and ComfyUI passes `allow_tf32=True` to its convolution explicitly, so
the patch-embed conv is not literally FP32 under either implementation; matmul
TF32 is off on both. Consistent, therefore not a divergence.

## The kernel axis: measured, and decided under a stated reading

**MEASURED.** [`h3_attention_kernel.py`](../../../../bench/h3_attention_kernel.py)
fixes whether SDPA sees grouped-query or expanded key/value heads. Codex's
four-step matrix, storage held at `comfy_exact_bf16_store`, relative L2 at
layer 49 on vision rows against deployed ComfyUI, with the accepted
`comfy_exact` residual in the grouped-math column
(`bench/results/2026-08-25_kernel_axis_{expansion,kernel,vs_comfy}_*.json`):

| fixture | grouped math (accepted) | expanded math | expanded efficient |
|---|---:|---:|---:|
| single-image 44x40 | 5.9e-2 | 2.2e-2 | 9.0e-2 |
| multi-image 18x18 x2 | 4.7e-4 | 4.4e-4 | 3.5e-4 |
| keyframe 48x84 | 2.9e-3 | 3.4e-3 | 2.9e-3 |
| mixed 84x48 + 22x22 | 5.1e-2 | 1.1e-1 | 5.8e-2 |

The expansion is not numerically free: on the two stable fixtures it and the
kernel arithmetic each move the state by the same few 1e-4 to 1e-3 as the
accepted residual. Text rows stay at 1e-4 to 1e-3 under every arm.

**MEASURED, the early-tap control.** On the two sensitive fixtures at decoder
layer 24, past the DeepStack injections, the three arms sit at 9.6e-4 /
1.35e-3 / 9.5e-4 (single) and 1.24e-3 / 2.01e-3 / 1.23e-3 (mixed) from
ComfyUI, with expanded-efficient indistinguishable from grouped math; by layer
49 the same arms are at percent level and their order has changed
(`2026-08-25_kernel_axis_tap24_*.json`). The layer-49 spread on those two
fixtures is compounding through the last 25 layers, not a kernel ranking.
Tap 0 is not a valid vision-row comparison point on this instrument, because
Transformers' hook reads before the DeepStack injection and ComfyUI's
truncated stack after it; any early tap must sit past layer 2.

**MEASURED, the composed path.** BF16 storage plus expanded-KV efficient
attention runs the entire five-row Gate 2A primary population (25,250 tokens,
longest row 10,358) through the real sequential path at 6.4 GiB allocated /
7.7 GiB reserved, and the 8,981-token stress row at 5.7 GiB, with the
10,358-token row's warm transient at 3.4 GiB where the math path would
allocate 25.6 (`2026-08-25_gate2a_primary_bf16_store_expanded_kv.json`,
`2026-08-25_gate2a_stress_bf16_store_expanded_kv.json`).

**ACTING-POINT DECISION.** Expanded-KV efficient attention is the Gate 2B
calibration kernel, under the mid-stack reading: at a tap past the DeepStack
injections it is indistinguishable from the accepted grouped-query math on
every fixture, and at layer 49 it is within the accepted residual's band on the
fixtures that can rank kernels. The alternative readings are recorded: applied
at layer 49 on all four fixtures, it is worse than grouped math by a factor of
1.1 to 1.5 on the two compounding fixtures. This is a calibration execution
policy, not bitwise parity, and it carries two conditions:

1. Gate 2B includes a **kernel-sensitivity control on the observer**: the AWQ
   scales for the first layers under grouped-query math and under expanded-KV
   efficient attention, on a row that fits both, compared directly. The
   layer-49 state is one row's endpoint; the calibration statistic is an
   aggregate, and whether it is sensitive at this level is **UNKNOWN** until
   that control runs.
2. Gate 5's numerical acceptance of the candidate against BF16 is unchanged by
   this decision and is where the kernel choice is ultimately answerable.

## Gate 2B entry contract, as now specified

Every item is measured or decided above; nothing is inherited from the
FP32-resident diagnostic.

1. Fresh full model per modifier arm, all 64 decoder layers, loaded through
   the supported converted offload bridge under
   `storage_policy(..., "comfy_exact_bf16_store")`.
2. `comfy_exact_bf16_store` precision policy, `expanded_kv` attention, the
   effective-input mask transform, all through the same pilot harness that
   produced the floors, so the increment is measured against them.
3. An explicit host-memory reserve passed to the bridge rather than its 5 GB
   default. The floor leaves 118 GiB available with 62 GiB of weights mapped;
   the reserve is a declared number in the run record, chosen so that the
   modifier's observer state on CPU and the intermediates cache (46 to 69 KB
   per token at peak) are budgeted, not assumed.
4. Recipe constructed in the pinned environment before the model loads, AWQ
   observer offload explicitly on CPU, the decoder-only target boundary
   checked before execution (embedding, vision tower and DeepStack stay BF16).
5. Measured against the composed-path floor: modifier state, cache/replay,
   cumulative CUDA, host RSS and `MemAvailable`, staging, time by stage, and
   cleanup.
6. An intentional abort, a mutation proving the modifier path was entered, and
   the kernel-sensitivity control above.
7. No candidate output path, serializer, symlink action, or publishing step.
8. Population: the Gate 2A primary rows. The interpolated stress row is not a
   calibration row and is not in the arm.

Gate 2B sets the row envelope, visual-block envelope, population/cache budget,
runtime budget and host reserve as separate dimensions. It may fix an absolute
population only if its measured arm covers the accepted workload shapes and
leaves a declared operating reserve.

## Operational rule

**OWNER-DECISION, 2026-08-25.** ComfyUI is stopped by its port owner before
any bridge load and restarted from `start.sh` afterwards. `/free` is not a
host-memory boundary: the first FP32 bridge load of the day peaked near total
system memory and the server exited during it.

## What this record does not establish

- Anything about the AWQ increment in memory, time or host state.
- Acceptance of expanded-KV attention as anything other than a calibration
  execution policy under the reading stated.
- Any render, fidelity or quality claim.
- An absolute calibration population.
