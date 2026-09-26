# Pipeline telemetry

Last updated: 2026-09-26

What a ComfyUI prompt loaded, moved and spent, end to end, as one JSONL
record per prompt. The recorder is `pipeline_telemetry.py`; the reader is
`bench/telemetry_report.py`; the controls are
`bench/check_pipeline_telemetry.py`. Asked for by the owner on 2026-09-26:
"we should know whats happening throughout our pipeline in a structured
schema and be able to record it end to end".

## Arming

The server reads its environment once, at pack import:

    H3_TELEMETRY=dir=<path>[,interval_ms=250][,blocks=1]

Unset, the module registers nothing. Armed, every prompt writes
`<dir>/<date>_<time>_<prompt id prefix>.jsonl`. It is the fourth arming key
beside the three in [`comfy_notes.md`](comfy_notes.md). Read the port
owner's environment before trusting a number, as for those. What arming
costs a render is not yet measured; until it is, compare a timing only with
another timing from a server armed the same way.

## How it sees, and what it cannot see

Nothing here patches core. The three sources:

1. **Node boundaries**, through core's cache-provider API
   (`comfy_execution/cache_provider.py`). On a local cache miss, just before
   a node executes, core calls the provider's `should_cache(context)`. When
   the node's output is stored, it calls `should_cache(context, value)`.
   Both calls are synchronous on the executor thread. The recorder snapshots
   there and returns False, so core never schedules a lookup or store.
   `never_caches` in the check holds that.
2. **A sampler thread**, every `interval_ms`. It reads NVML through ctypes
   (the device is matched by CUDA UUID), dynamic VRAM's total, `/proc/self`
   and `/proc/meminfo`. It makes only reads that are safe from another
   thread; per-page residency is read at boundaries only.
3. **Core's log lines** about loading, staging, unloading, pin eviction,
   AIMDO frees and OOM fallbacks, parsed by `_LOAD_PATTERNS`. The recorder
   lowers the root logger to DETAIL (15) so those lines reach its handler.
   The console keeps its own level.

**Blind spots, by construction:**

- **A node core serves from its cache** never calls `should_cache`, so it has
  no boundary. Neither does a node whose cache key cannot be serialised or
  that contains a self-unequal value (`comfy_execution/caching.py`,
  `_check_providers_lookup`). The reader lists every such node under
  `not_observed` rather than guessing which case applies. `/history`'s
  `execution_cached` message names the cache hits.
- **Individual transfers and evictions.** Dynamic VRAM's native library has
  no counters and the cast path has no callback. What the record has is the
  PCIe rate from NVML and residency before and after each node.
- **Per-sampler-step detail.** Steps fall inside the sampler node's window.
  Samples carry their time, so a step boundary can be read against them, but
  the record does not mark steps.

## Schema, version 1

Every row has `t` (seconds since the prompt started, monotonic), `seq`, and
`type`.

- **`header`:** once per record.
  - `schema`, `prompt_id`, `wall_start`, `pid`, `spec`.
  - `substrate`: the GPU's name, driver, maximum PCIe generation and width
    and power limit; torch and CUDA versions; the server's argv; ComfyUI's
    and the pack's commits; the `comfy_aimdo` and `comfy_kitchen` versions;
    whether dynamic VRAM is on; host RAM and swap.
  - `graph`: the sha256 of the prompt's canonical JSON, and every node's
    class.
- **`prompt_start`, `node_start`, `node_end`, `prompt_end`:** a snapshot.
  - `node` and `class_type` (empty on the prompt rows).
  - `device`: torch's free, total, allocated, reserved and peak-allocated
    bytes; dynamic VRAM's total VRAM use (`aimdo_vram`); pinned host memory
    in total.
  - `models`, one entry per entry in `current_loaded_models`:
    - class, patcher class, `dynamic`, device, `currently_used`;
    - `size`, `loaded` (bytes on the card), `ram_loaded`, `pinned`,
      `force_loaded`;
    - for a dynamic model, `vbar`: page count, resident and pinned pages,
      watermark, loaded bytes. With `blocks=1` it also carries
      `block_resident`, the fraction of each `blocks.N`'s pages that are
      resident.
- **`sample`:** every interval.
  - `node`: the node executing then, or null.
  - `gpu`: memory used and total; utilisation of GPU and memory; `pcie_tx_kbs`
    (card to host) and `pcie_rx_kbs` (host to card), each NVML's short-window
    rate; current PCIe generation and width; power; SM and memory clocks;
    temperature.
  - `host`: the process's RSS, split into anonymous, file-backed and shared,
    plus swap and threads; its disk read and write bytes and `rchar`; its user
    and system CPU seconds. The host's total, free and available memory, page
    cache, dirty and writeback bytes, swap, mlocked and unevictable memory,
    and major faults.
  - `aimdo_vram`: dynamic VRAM's total.
- **`log`:** one per matched core log line. It carries `kind` (the pattern's
  name), `groups` (the parsed fields), `node` (the node executing then), and
  the logger, level and message.

**What the fields mean under dynamic VRAM**, from reading core and
`comfy_aimdo` on 2026-09-26:

- **"Staged" is address space, not data.** It is a virtual reservation on
  the card, and nothing is resident when the line prints.
- **Weights move in 32 MB pages when a layer runs.** Each page comes from a
  pinned host copy or from the memory-mapped `.safetensors` file. Pages
  become evictable after use.
- **So `proc_rssfile` and the page cache are the file-backed weights.**
  `proc_rssanon` includes the pinned copies.
- **torch's allocator figures likely exclude dynamic VRAM's pages.**
  `aimdo_vram` and NVML's `used` are the totals to read. This is inferred
  from the code and not yet confirmed against a record.

## Reading a record

    python bench/telemetry_report.py <record>.jsonl [--json summary.json]

It prints one row per observed node: seconds, peak card memory by NVML and
by dynamic VRAM, estimated PCIe gigabytes each way, mean utilisation, the
change in anonymous memory and page cache, disk reads, major faults and CPU
seconds. Beneath each row are the models at its start and end, with bytes on
the card, bytes pinned and the mean DiT block residency. Byte totals are
trapezoids over the samples; a node shorter than two samples reports `None`.
