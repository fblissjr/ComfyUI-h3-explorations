#!/usr/bin/env python3
"""Gate 2A: the sequential floor envelope, measured, on one 4090.

This drives `llm-compressor`'s actual `SequentialPipeline` over real native-H3
batches and records the resource curve. It **does not quantize**: the session is
initialised with no modifiers, so the pipeline traces the model, materialises
the intermediates cache, and runs every subgraph forward -- including the
error-propagation replay -- while no modifier hook observes or rewrites a single
weight.

**This is a floor, and it cannot set the calibration population.** The expensive
parts measured here -- tracing, the cache, two forwards per subgraph per row --
are the ones a calibration run pays before AWQ does anything. AWQ adds
activation observation, its smoothing search, and in-memory weight rewriting on
top, and none of that is in these numbers. Setting a population budget needs
Gate 2B: a bounded, disposable, modifier-bearing run that measures the AWQ
increment. Reading this file's output as a budget would understate the real
cost by an unmeasured margin.

**A pilot that emitted a checkpoint would be a launch**, and this one has
nothing to emit: no save is called and no output directory is created, under
either mode below.

**Gate 2B mode** (`--modifier awq`) instantiates the real v2 recipe from
`bench/h3_awq_recipe.py` -- the AWQ modifier with its activation cache on the
CPU, and the W4A16 quantization modifier bounded to the decoder linears -- and
runs the same sequential path with the modifier observing, searching and
rewriting weights in the offload store. It measures the increment over the
floor: smoothing time, parent re-runs per mapping, scale records per mapping
(saved beside the report so two arms can be compared), the modifier's cache
placement, and a control that a balance-layer weight actually changed in the
store. One prefix per process, because AWQ mutates the weights and a second
prefix would calibrate on already-smoothed ones. The boundary is asserted after
the session applies the config and before any forward: exactly the text
decoder linears carry a weight scheme, nothing in the tower, mergers, embedding
or head.

It answers, by measuring:

- peak allocated and reserved VRAM, cumulative over cache construction and
  every subgraph forward, with per-call entry residency and transient growth;
- host RAM three ways: the kernel's high-water mark, current RSS, and what the
  system still reports available, before the load, after it, and per step;
- where the intermediates cache lives and how it grows, sampled after every
  forward rather than read off the residual at the end;
- whether the replay pass really runs, counted at `Subgraph.forward`;
- time by observable stage: trace, cache build, subgraph forwards;
- that the declared all-ones-mask omission survives the real dataloader, the
  cache and the traced graph, read from each of those objects;
- physical staging under the offload directory, symlinks deduplicated; and
- what is left behind after completion, a deliberate abort, and an OOM.

And the question the full-forward OOM cannot answer: **whether an active
subgraph can be promoted to FP32 while the inactive weights stay BF16 and
offloaded.** A full FP32 forward needs the whole stack resident at FP32; the
sequential path needs one subgraph at a time, so the earlier OOM does not
predict this and is not allowed to stand in for it.

The population escalates over selected prefixes so a growth curve is visible
without paying for every one: `--prefix` defaults to 1, 3 and the full set,
which is three pipeline runs over five rows rather than fifteen row-instances.
Order the rows small to large -- single image, mixed keyframe/reference,
multi-image, genuine reference video, and the separately named 2048-upscale
stress row -- so a step that does not fit stops the escalation at a known point.
That stop is the measurement, not a failure.

Run it with the `llm-compressor` virtualenv python and a free GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import platform
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file
from torch.utils.data import DataLoader

import llmcompressor
from llmcompressor.args import DatasetArguments
from llmcompressor.core import create_session
from llmcompressor.datasets.utils import get_calibration_dataloader
from llmcompressor.pipelines.sequential import pipeline as sequential_pipeline
from llmcompressor.pipelines.sequential.helpers import Subgraph
from safetensors.torch import save_file

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
REPORT = BENCH / "results" / "2026-08-24_sequential_feasibility_pilot.json"

from h3_calibration_precision import (  # noqa: E402
    POLICIES,
    POLICY_INTENT,
    calibration_precision,
    compute_dtype,
    storage_dtype,
    storage_policy,
)
from h3_attention_kernel import ATTENTION_KINDS, attention_kernel  # noqa: E402
from h3_effective_batch import effective_batch, tensor_sha  # noqa: E402
from h3_producer_provenance import producer_provenance  # noqa: E402

SEQUENTIAL_TARGETS = ["Qwen3VLTextDecoderLayer"]


# --------------------------------------------------------------------------
# measurement


def host_peak_kib() -> int:
    """The kernel's own high-water mark for this process, not a sampled guess."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _proc_kib(path: str, key: str) -> int | None:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(key):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def host_memory() -> dict:
    """Current occupancy and what the system says is still available.

    `ru_maxrss` is a *historical* high-water mark: a peak of 121 GiB does not
    say that only a few GiB were free when the next stage would begin, because
    the peak may have been released long before. Current RSS and the kernel's
    own `MemAvailable` are the two that bound what a later stage can still ask
    for, so all three are reported and none of them stands in for the others.
    """
    return {
        "peak_rss_gib": round(host_peak_kib() / 2**20, 2),
        "current_rss_gib": (
            round(_proc_kib("/proc/self/status", "VmRSS:") / 2**20, 2)
            if _proc_kib("/proc/self/status", "VmRSS:") else None
        ),
        "system_mem_available_gib": (
            round(_proc_kib("/proc/meminfo", "MemAvailable:") / 2**20, 2)
            if _proc_kib("/proc/meminfo", "MemAvailable:") else None
        ),
    }


def cuda_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def cuda_peak() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }


class _CumulativePeak:
    """A step-wide maximum that survives the per-subgraph stat resets.

    The per-subgraph split needs `reset_peak_memory_stats` before every forward,
    which makes torch's own peak describe only the last window. Reading that at
    the end of a step would report the final subgraph's peak as the step's --
    an under-report with no warning attached. This keeps the running maximum.
    """

    def __init__(self) -> None:
        self.allocated = 0.0
        self.reserved = 0.0

    def observe(self, peak: dict | None = None) -> None:
        peak = peak if peak is not None else cuda_peak()
        self.allocated = max(self.allocated, peak.get("allocated_gib", 0.0))
        self.reserved = max(self.reserved, peak.get("reserved_gib", 0.0))

    def as_dict(self) -> dict:
        return {"allocated_gib": round(self.allocated, 3),
                "reserved_gib": round(self.reserved, 3),
                "note": "cumulative across cache construction and every "
                        "subgraph forward, not the last reset window"}


def directory_bytes(path: Path) -> dict:
    """Physical staging bytes, with symlinked targets counted separately.

    `stat()` follows symlinks, so an offload directory that links to the source
    checkpoint would report tens of gigabytes of "temporary disk use" that were
    never written. `lstat` measures what is actually staged; the link targets
    are reported beside it as a logical figure that is explicitly not disk
    consumption.
    """
    physical = links = files = 0
    targets: dict[str, int] = {}
    for entry in path.rglob("*"):
        try:
            info = entry.lstat()
        except OSError:
            continue
        if entry.is_symlink():
            links += 1
            physical += info.st_size
            # Deduplicate by resolved target. Hundreds of links can point into
            # one checkpoint shard, and summing per link would report that shard
            # once per link -- a number larger than the disk holds.
            with contextlib.suppress(OSError):
                resolved = entry.resolve()
                targets[str(resolved)] = resolved.stat().st_size
        elif entry.is_file():
            files += 1
            physical += info.st_size
    return {"physical_bytes": physical, "files": files, "symlinks": links,
            "unique_symlink_targets": len(targets),
            "unique_symlink_target_bytes_not_staged": sum(targets.values())}


def _cache_bytes(cache) -> tuple[int, dict[str, int]]:
    """Tensor count and bytes by device, walked without moving anything."""
    by_device: dict[str, int] = {}
    tensors = 0

    def walk(value):
        nonlocal tensors
        inner = getattr(value, "value", value)
        if torch.is_tensor(inner):
            tensors += 1
            key = str(inner.device)
            by_device[key] = by_device.get(key, 0) + inner.numel() * inner.element_size()
        elif isinstance(inner, (list, tuple)):
            for item in inner:
                walk(item)
        elif isinstance(inner, dict):
            for item in inner.values():
                walk(item)

    for batch in cache.batch_intermediates:
        for value in batch.values():
            walk(value)
    return tensors, by_device


def cache_keys(cache) -> list[str]:
    """Every key any batch currently holds. The effective-input proof reads this."""
    keys: set[str] = set()
    for batch in cache.batch_intermediates:
        keys.update(batch.keys())
    return sorted(keys)


def cache_footprint(cache) -> dict:
    """Where the intermediates cache actually lives, and how large it is.

    Read off the cache's own structure rather than inferred from row count:
    `IntermediatesCache` stores an `IntermediateValue` per key per batch, and
    the offload device is a property of how it was constructed, so a run that
    silently kept activations on the accelerator would show up here.

    **A footprint read after the run is the residual, not the growth.** The
    pipeline deletes each subgraph's consumed inputs as it goes, so what is
    left at the end is the last subgraph's inputs. The peak is sampled inside
    the forward wrapper instead and reported beside this.
    """
    tensors, by_device = _cache_bytes(cache)
    return {
        "tensors": tensors,
        "bytes_by_device": by_device,
        "gib_by_device": {k: round(v / 2**30, 3) for k, v in by_device.items()},
        "declared_offload_device": str(cache.offload_device),
        "keys": cache_keys(cache),
    }


# --------------------------------------------------------------------------
# the population


class _RowDataset(torch.utils.data.Dataset):
    """Effective batches for a chosen prefix of the bundle's declared order."""

    def __init__(self, bundle: Path, manifest: dict, row_ids: list[str]):
        self.bundle = bundle
        self.records = {r["row_id"]: r for r in manifest["rows"]}
        self.order = list(row_ids)
        self.transforms: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index: int) -> dict:
        row_id = self.order[index]
        raw = load_file(self.bundle / self.records[row_id]["batch_file"])
        batch, transform = effective_batch(raw, row_id=row_id)
        self.transforms[row_id] = transform
        return batch


def _identity_collate(rows: list[dict]) -> dict:
    if len(rows) != 1:
        raise ValueError("this pilot is one row per batch; see prove_calibration_seam.py")
    return rows[0]


def build_loader(bundle: Path, manifest: dict, row_ids: list[str]) -> DataLoader:
    dataset = _RowDataset(bundle, manifest, row_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_identity_collate)
    returned = get_calibration_dataloader(DatasetArguments(dataset=loader), processor=None)
    if returned is not loader:
        raise AssertionError("the library did not return the preconstructed DataLoader")
    return returned


# --------------------------------------------------------------------------
# instrumentation of the real pipeline


class _Instrumentation:
    """Counters and timers wrapped around the pipeline's own call sites.

    Wrapping `Subgraph.forward` and `IntermediatesCache.from_dataloader` means
    the numbers come from the code that actually runs, not from a model of it.
    The replay count in particular cannot be asserted from `propagate_error`
    alone -- that flag says what was requested; this counts what happened.
    """

    def __init__(self) -> None:
        self.forward_calls = 0
        self.forward_seconds = 0.0
        self.cache = None
        self.cache_build_seconds = 0.0
        self.subgraph_peak: dict[int, dict] = {}
        self.peak: "_CumulativePeak | None" = None
        self.seen_subgraphs: set[int] = set()
        self.oom_message: str | None = None
        self.oom_stage: str | None = None
        self.oom_on_cold_call: bool | None = None
        self.abort_after: int | None = None
        self.trace_seconds = 0.0
        self.trace: dict | None = None
        self.cache_initial: dict | None = None
        self.cache_keys_at_first_forward: list[str] | None = None
        self.cache_peak_bytes: dict[str, int] = {}
        self.cache_peak_tensors = 0
        self.awq_smoothing_seconds = 0.0
        self.awq_smoothing_calls = 0
        self.awq_parent_runs = 0
        self.awq_scales: list[dict] = []
        self.awq_scale_tensors: dict[str, torch.Tensor] = {}

    @contextlib.contextmanager
    def instrument_awq(self, modifier):
        """Count and time what the AWQ modifier actually does, on its own methods.

        `_apply_smoothing` is what the sequential pipeline calls at the end of
        each subgraph's calibration pass; `_run_samples` is the parent-module
        re-run the grid search pays once per grid point; `_compute_best_scale`
        returns the per-channel scales for one mapping. Wrapping the bound
        methods on this instance leaves the class alone and gives the count of
        parent re-runs -- the AWQ multiplier over the floor -- as a measurement
        rather than a source-read estimate.
        """
        instrumentation = self
        original_smooth = modifier._apply_smoothing
        original_runs = modifier._run_samples
        original_scale = modifier._compute_best_scale

        def apply_smoothing(model):
            started = time.time()
            try:
                return original_smooth(model)
            finally:
                instrumentation.awq_smoothing_seconds += time.time() - started
                instrumentation.awq_smoothing_calls += 1
                instrumentation._sample_cache()

        def run_samples(module):
            instrumentation.awq_parent_runs += 1
            return original_runs(module)

        def compute_best_scale(mapping, fp16_outputs, orig_layer_weights):
            scales = original_scale(mapping, fp16_outputs, orig_layer_weights)
            flat = scales.detach().float().cpu().contiguous()
            key = mapping.smooth_name
            instrumentation.awq_scale_tensors[key] = flat
            instrumentation.awq_scales.append({
                "smooth": key,
                "balance": list(getattr(mapping, "balance_names", [])),
                "channels": int(flat.numel()),
                "mean": float(flat.mean()), "min": float(flat.min()),
                "max": float(flat.max()),
                "sha256": tensor_sha(flat),
            })
            return scales

        modifier._apply_smoothing = apply_smoothing
        modifier._run_samples = run_samples
        modifier._compute_best_scale = compute_best_scale
        try:
            yield
        finally:
            modifier._apply_smoothing = original_smooth
            modifier._run_samples = original_runs
            modifier._compute_best_scale = original_scale

    def _sample_cache(self) -> None:
        """Running maximum of the cache's bytes by device, one walk per call.

        Cheap: a handful of tensors per row and no data movement. This is the
        only way to see growth, because the pipeline deletes consumed inputs
        as it goes and the end-of-run footprint is a residual.
        """
        if self.cache is None:
            return
        tensors, by_device = _cache_bytes(self.cache)
        self.cache_peak_tensors = max(self.cache_peak_tensors, tensors)
        for device, size in by_device.items():
            self.cache_peak_bytes[device] = max(self.cache_peak_bytes.get(device, 0), size)

    @contextlib.contextmanager
    def install(self):
        original_forward = Subgraph.forward
        original_cache = sequential_pipeline.IntermediatesCache.from_dataloader
        original_trace = sequential_pipeline.trace_subgraphs
        instrumentation = self

        def trace_subgraphs(model, sample_input, *args, **kwargs):
            # The trace is its own observable stage, and the one place the
            # effective-input rule can be checked against the graph `oneshot`
            # executes: a key absent from the sample batch becomes a constant
            # with no placeholder, so `attention_mask` must not be an input
            # of any subgraph, and `pixel_values` must be one of the first.
            started = time.time()
            subgraphs = original_trace(model, sample_input, *args, **kwargs)
            instrumentation.trace_seconds = time.time() - started
            names = [sorted(s.input_names) for s in subgraphs]
            instrumentation.trace = {
                "subgraphs": len(subgraphs),
                "sample_input_keys": (sorted(sample_input) if isinstance(sample_input, dict)
                                      else None),
                "first_subgraph_input_names": names[0] if names else [],
                "attention_mask_in_any_subgraph_inputs": any(
                    "attention_mask" in n for n in names
                ),
                "pixel_values_in_first_subgraph_inputs": (
                    "pixel_values" in names[0] if names else False
                ),
                "image_grid_thw_in_first_subgraph_inputs": (
                    "image_grid_thw" in names[0] if names else False
                ),
            }
            return subgraphs

        def forward(self, *args, **kwargs):
            if (instrumentation.abort_after is not None
                    and instrumentation.forward_calls >= instrumentation.abort_after):
                raise DeliberateAbort(
                    f"aborting after {instrumentation.forward_calls} subgraph forwards"
                )
            if instrumentation.forward_calls == 0 and instrumentation.cache is not None:
                # Before anything is consumed: the keys the cache handed the
                # first subgraph are the effective batch as `oneshot` saw it.
                instrumentation.cache_keys_at_first_forward = cache_keys(
                    instrumentation.cache
                )
            started = time.time()
            allocated_before = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            # First call for this subgraph versus a later one. A subgraph's
            # weights are onloaded lazily inside the forward, so the entry
            # value on a cold call does not yet include them and cannot be
            # read as weight residency. Only the warmed calls can be.
            key = id(self)
            cold = key not in instrumentation.seen_subgraphs
            instrumentation.seen_subgraphs.add(key)
            try:
                return original_forward(self, *args, **kwargs)
            except torch.OutOfMemoryError as exc:
                # Keep the allocator's own words. The outer
                # `handle_sequential_oom` wrapper replaces this with generic
                # advice, and `max_memory_allocated()` cannot report the size of
                # an allocation that never succeeded -- which is why an OOM on a
                # large row can show a *lower* recorded peak than one on a
                # smaller row that got further.
                instrumentation.oom_message = str(exc).strip().splitlines()[0][:400]
                instrumentation.oom_stage = "subgraph_forward"
                instrumentation.oom_on_cold_call = cold
                raise
            finally:
                instrumentation.forward_calls += 1
                instrumentation.forward_seconds += time.time() - started
                instrumentation._sample_cache()
                peak = cuda_peak()
                if instrumentation.peak is not None:
                    instrumentation.peak.observe(peak)
                instrumentation.subgraph_peak[instrumentation.forward_calls] = {
                    **peak,
                    "cold_call": cold,
                    "allocated_before_gib": round(allocated_before / 2**30, 3),
                    "transient_gib": round(
                        max(0.0, peak.get("allocated_gib", 0.0)
                            - allocated_before / 2**30), 3
                    ),
                }

        # `original_cache` is already a bound classmethod, and after the
        # library's own decoration it can be a `functools.partial` with no
        # `__func__`. Calling the bound object is the form that works whatever
        # it is wrapped in; reaching for `__func__` broke on the first real run.
        def from_dataloader(dataloader, model_device, offload_device):
            started = time.time()
            try:
                cache = original_cache(dataloader, model_device, offload_device)
            except torch.OutOfMemoryError as exc:
                # The same capture as in the forward wrapper, for the same
                # reason. `IntermediatesCache.from_dataloader` materialises
                # every row before the first subgraph runs, so a large-row
                # population can fail here -- before any `Subgraph.forward` is
                # reached, which is the only other place the allocator's own
                # message is preserved. Without this, a cache-stage OOM reaches
                # the report as the pipeline wrapper's generic advice and the
                # failed allocation size is lost.
                instrumentation.oom_message = str(exc).strip().splitlines()[0][:400]
                instrumentation.oom_stage = "intermediates_cache_build"
                if instrumentation.peak is not None:
                    instrumentation.peak.observe()
                raise
            instrumentation.cache_build_seconds = time.time() - started
            if instrumentation.peak is not None:
                instrumentation.peak.observe()
            instrumentation.cache = cache
            instrumentation.cache_initial = cache_footprint(cache)
            instrumentation._sample_cache()
            return cache

        Subgraph.forward = forward
        sequential_pipeline.IntermediatesCache.from_dataloader = staticmethod(from_dataloader)
        sequential_pipeline.trace_subgraphs = trace_subgraphs
        try:
            yield self
        finally:
            Subgraph.forward = original_forward
            sequential_pipeline.IntermediatesCache.from_dataloader = original_cache
            sequential_pipeline.trace_subgraphs = original_trace


class DeliberateAbort(RuntimeError):
    """The controlled failure. Raised from inside a subgraph forward."""


# --------------------------------------------------------------------------


def load_model(source: Path, policy: str, layers: int, gpu_gib: float,
               offload_dir: Path, offload: str, host_reserve_gib: float | None = None):
    """Load the calibration model under one of two offload arrangements.

    `host` loads plainly on the CPU and lets `SequentialPipeline` move each
    subgraph to the accelerator itself. Every parameter is resident in host RAM,
    which at FP32 over 50 layers is most of this box.

    `auto_offload` uses the official bridge: `llmcompressor.utils.dev
    ::load_context` wraps `compressed_tensors.offload::load_offloaded_model`,
    which loads through Accelerate and then calls `from_accelerate` to *replace*
    Accelerate's hooks with compressed-tensors offload caches. `auto_offload`
    additionally restricts placement to CPU and disk, so anything that does not
    fit in host memory spills to the offload directory.

    **The distinction matters and an earlier version of this file got it
    wrong.** What was measured is that *raw, unconverted* `device_map="auto"`
    does not compose with `SequentialPipeline`: Accelerate replaces each hooked
    module's `forward` with a `functools.partial`, and
    `compressed_tensors.offload.module.offload_module` -- reached through
    `set_onload_device` before the first batch -- reads
    `module.forward.__func__`. That failure is real and reproducible. It is not
    evidence about the conversion path, which exists precisely to remove those
    hooks, and this file previously generalised one to the other.
    """
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(source)
    config.text_config.num_hidden_layers = layers
    # Storage dtype, not compute dtype: they differ under the manual-cast
    # policy, and `storage_policy` is what keeps the patch embed in FP32 there.
    kwargs = {
        "config": config,
        "dtype": storage_dtype(policy),
        "attn_implementation": "sdpa",
    }
    if offload == "host":
        with storage_policy(Qwen3VLForConditionalGeneration, policy):
            return Qwen3VLForConditionalGeneration.from_pretrained(source, **kwargs).eval()

    from compressed_tensors.offload import load_offloaded_model
    from llmcompressor.modeling.moe.linearize import load_quantizable_moe

    # `load_context` is `load_offloaded_model` composed with
    # `load_quantizable_moe`, with the bridge's default host reserve
    # (`extra_cpu_mem`, 5 GB) for everything that is not model loading. The
    # same composition is used here so an explicit reserve can be passed: the
    # Gate 2B contract requires the reserve to be a declared number in the run
    # record, not the bridge default.
    reserve = None if host_reserve_gib is None else int(host_reserve_gib * 2**30)
    bridge = (load_offloaded_model(Qwen3VLForConditionalGeneration)
              if reserve is None else
              load_offloaded_model(Qwen3VLForConditionalGeneration, extra_cpu_mem=reserve))
    with storage_policy(Qwen3VLForConditionalGeneration, policy):
        with bridge, load_quantizable_moe(Qwen3VLForConditionalGeneration):
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                source, device_map="auto_offload", offload_folder=str(offload_dir),
                **kwargs,
            )
    return model.eval()


def offload_topology(model) -> dict:
    """Where the weights actually are, without onloading a single one.

    **Iterating `model.parameters()` on a disk-offloaded model onloads every
    parameter.** That is not a reporting detail: it pulls the whole checkpoint
    through host RAM, contaminates the load time and the RSS high-water mark,
    and destroys the very topology the field was meant to describe. An earlier
    version of this file did exactly that, so any host figure it produced under
    `auto_offload` describes the instrumentation as much as the bridge.

    `compressed_tensors` keeps offloaded tensors in an `OffloadCache` at
    `module._parameters.offloaded_values`, which can be read without onloading.
    This also asserts the conversion actually happened: an Accelerate hook or a
    surviving `hf_device_map` means the raw path is still in place, which is the
    arrangement `SequentialPipeline` cannot use.
    """
    from compressed_tensors.offload.cache import OffloadCache

    by_device: dict[str, int] = {}
    by_dtype: dict[str, int] = {}
    offloaded = resident = 0
    accelerate_hooks = []

    def record(where: str, tensor) -> None:
        nonlocal by_device, by_dtype
        if tensor is None:
            return
        by_device[where] = by_device.get(where, 0) + tensor.numel() * tensor.element_size()
        dt = str(getattr(tensor, "dtype", "?")).removeprefix("torch.")
        by_dtype[dt] = by_dtype.get(dt, 0) + 1

    for name, module in model.named_modules():
        for attribute in ("_parameters", "_buffers"):
            store = getattr(module, attribute, None)
            if isinstance(store, OffloadCache):
                # Key by the CACHE's offload device, not the tensor's. A
                # disk-offloaded tensor reports `meta` for its own device, so
                # keying on that would file every disk-resident weight under
                # `meta` and the report could not tell CPU from disk -- which is
                # the one thing this field exists to say.
                where = str(getattr(store, "offload_device", "unknown"))
                for tensor in store.offloaded_values.values():
                    if tensor is None:
                        continue
                    offloaded += 1
                    record(where, tensor)
            else:
                for tensor in (store or {}).values():
                    if tensor is None:
                        continue
                    resident += 1
                    record(str(tensor.device), tensor)
        if hasattr(module, "_hf_hook"):
            accelerate_hooks.append(name)

    return {
        "offloaded_tensors": offloaded,
        "resident_tensors": resident,
        "bytes_by_offload_device": by_device,
        "gib_by_offload_device": {k: round(v / 2**30, 2) for k, v in by_device.items()},
        "tensors_by_dtype": by_dtype,
        "accelerate_hooks_remaining": accelerate_hooks[:5],
        "accelerate_hook_count": len(accelerate_hooks),
        "hf_device_map_present": hasattr(model, "hf_device_map"),
        "conversion_clean": not accelerate_hooks and not hasattr(model, "hf_device_map"),
        "summary": (f"{offloaded} offloaded + {resident} resident tensors, "
                    f"dtypes {sorted(by_dtype)}, "
                    f"{ {k: round(v / 2**30, 1) for k, v in by_device.items()} }"),
        "read_without_onloading": True,
    }


def _probe_backends(query, key, is_causal: bool, gqa: bool) -> dict:
    from torch.backends.cuda import (
        SDPAParams,
        can_use_cudnn_attention,
        can_use_efficient_attention,
        can_use_flash_attention,
    )

    params = SDPAParams(query, key, key, None, 0.0, is_causal, gqa)
    return {
        "flash": bool(can_use_flash_attention(params, False)),
        "efficient": bool(can_use_efficient_attention(params, False)),
        "cudnn": bool(can_use_cudnn_attention(params, False)),
    }


def sdpa_availability_for_rows(records: list[dict], config,
                               dtype: torch.dtype) -> dict:
    """Which fused SDPA backends exist for the shapes this step actually uses.

    **Probed at the real shapes, not a synthetic one.** Availability can depend
    on sequence length and on `is_causal`, so a 512-token causal text probe
    cannot be paired with an 8,981-token language footprint or with a
    128x128 vision block -- an earlier version of this function did exactly
    that, and its result had no standing in an OOM attribution.

    Text attention is probed at each row's own sequence length, in both the
    grouped-query form the model declares and the expanded form a `repeat_kv`
    would produce, causal. Vision attention is probed at each unique block
    patch count, non-causal and without grouped-query, because the tower
    attends fully within a block.

    Availability, not selection: this API reports what could run, not what was
    selected. A forward cannot have used a backend that is unavailable, and
    that is the whole of what this bounds. Identifying the kernel that actually
    ran is `bench/probe_sdpa_backend_selection.py`, which names the dispatched
    operation with `torch.profiler` and is kept separate so its overhead cannot
    contaminate these feasibility numbers.
    """
    if not torch.cuda.is_available():
        return {"available": None, "reason": "no CUDA"}

    text = config.text_config
    heads, kv_heads, head_dim = (text.num_attention_heads,
                                 text.num_key_value_heads, text.head_dim)
    vision = config.vision_config
    vision_heads = vision.num_heads
    vision_head_dim = vision.hidden_size // vision_heads

    lengths = sorted({r["sequence_length"] for r in records})
    patch_counts = sorted({
        int(b["grid_thw"][0][1]) * int(b["grid_thw"][0][2])
        for r in records for b in r["vision_blocks"]
    })

    out: dict = {"text_attention": {}, "vision_attention": {},
                 "probed_at_real_row_shapes": True,
                 "note": "availability, not selection: this API reports what "
                         "could run. probe_sdpa_backend_selection.py names "
                         "what did"}
    for length in lengths:
        q = torch.zeros(1, heads, length, head_dim, dtype=dtype, device="cuda")
        grouped = torch.zeros(1, kv_heads, length, head_dim, dtype=dtype, device="cuda")
        expanded = torch.zeros(1, heads, length, head_dim, dtype=dtype, device="cuda")
        out["text_attention"][str(length)] = {
            "sequence_tokens": length, "is_causal": True,
            "grouped_query": _probe_backends(q, grouped, True, True),
            "expanded_kv": _probe_backends(q, expanded, True, False),
        }
        del q, grouped, expanded
    for patches in patch_counts:
        q = torch.zeros(1, vision_heads, patches, vision_head_dim,
                        dtype=dtype, device="cuda")
        out["vision_attention"][str(patches)] = {
            "block_patches": patches, "is_causal": False,
            "full_attention": _probe_backends(q, q, False, False),
        }
        del q
    torch.cuda.empty_cache()
    return out


ATTRIBUTION_CAVEAT = (
    "A nominal footprint matching the captured allocator request supports an "
    "attribution; it does not prove one. These are single-tensor sizes, not "
    "predicted allocator requests: the selected SDPA backend may allocate "
    "additional buffers, choose a different internal layout, or split the "
    "computation -- cuDNN and memory-efficient attention need workspace too, "
    "and which backend runs is deliberately still unknown here. "
    "Read this only together with the captured allocator message, the failure "
    "stage, and the backend availability above."
)


def nominal_attention_logit_footprints(record: dict, config,
                                       dtype: torch.dtype) -> dict:
    """Nominal one-attention-logit tensor footprints for one row.

    Two quadratics matter and they are separate: the language stack attends
    over the row's whole sequence, and the vision tower attends within each
    vision block, so the largest single block governs the second. Neither is a
    function of the population's total tokens, because rows are processed one
    at a time.
    """
    element = torch.finfo(dtype).bits // 8
    heads = config.text_config.num_attention_heads
    vision_heads = config.vision_config.num_heads
    tokens = record["sequence_length"]
    grids = [[int(b["grid_thw"][0][1]), int(b["grid_thw"][0][2])]
             for b in record["vision_blocks"]]
    patches = [h * w for h, w in grids]
    largest = max(patches, default=0)
    return {
        "kind": "nominal one-attention-logit tensor footprints",
        "is_a_predicted_allocator_request": False,
        "caveat": ATTRIBUTION_CAVEAT,
        "element_bytes": element,
        "language": {
            "heads": heads,
            "sequence_tokens": tokens,
            "nominal_gib": round(heads * tokens * tokens * element / 2**30, 2),
        },
        "vision": {
            "heads": vision_heads,
            "block_grids": grids,
            "block_patches": patches,
            "largest_block_patches": largest,
            "nominal_gib": round(
                vision_heads * largest * largest * element / 2**30, 2
            ),
        },
    }


def effective_input_record(transforms: dict[str, dict], row_ids: list[str],
                           instrumentation: "_Instrumentation") -> dict:
    """Did the declared mask omission survive the real dataloader, cache and trace?

    `active_plan.md` Gate 2A: "the pilot must also prove that the effective
    all-ones-mask omission survives the real sequential dataloader and cache
    path". Three independent observation points, each read from the object
    that owns it rather than from the transform's own record:

    1. the transform record every `__getitem__` produced -- the assertion that
       the raw mask was all ones, and both hashes;
    2. the keys the `IntermediatesCache` held when the first subgraph ran; and
    3. the placeholder names of every traced subgraph.

    `attention_mask` must be absent from 2 and 3, and every row in 1 must have
    passed its assertion. A run that rebuilt the mask somewhere between the
    dataset and the graph would show it at 2 or 3.
    """
    rows = {}
    for row_id in row_ids:
        record = transforms.get(row_id)
        rows[row_id] = None if record is None else {
            "transform": record["transform"],
            "assertion_holds": record["assertion_holds"],
            "attention_mask_non_one_elements": record["attention_mask_non_one_elements"],
            "raw_presentation_sha256": record["raw_presentation_sha256"],
            "effective_model_input_sha256": record["effective_model_input_sha256"],
            "omitted_keys": record["omitted_keys"],
            "effective_keys": record["effective_keys"],
        }
    cache_keys_seen = instrumentation.cache_keys_at_first_forward
    trace = instrumentation.trace
    in_cache = (None if cache_keys_seen is None
                else "attention_mask" in cache_keys_seen)
    in_trace = (None if trace is None
                else trace["attention_mask_in_any_subgraph_inputs"])
    every_row = all(r is not None and r["assertion_holds"] for r in rows.values())
    return {
        "rows": rows,
        "every_row_transformed_with_assertion": every_row,
        "cache_keys_at_first_forward": cache_keys_seen,
        "attention_mask_in_cache_at_first_forward": in_cache,
        "attention_mask_in_traced_subgraph_inputs": in_trace,
        "omission_survives_dataloader_cache_and_trace": (
            every_row and in_cache is False and in_trace is False
        ),
        "note": "None means that observation point was never reached in this "
                "step, which is not the same as the mask being absent",
    }


def _offloaded_weight_location(module) -> dict:
    """Where a weight sits in the offload store: its tier and, on disk, its file.

    The CPU tier holds the tensor itself; the disk tier holds a meta tensor
    whose data is in the file its cache's index names. That file is a symlink
    into the checkpoint until the first update, which unlinks it and writes a
    staged file in the offload directory, so `staged` is the observable that a
    disk-tier weight has been rewritten.
    """
    from compressed_tensors.offload.cache import DiskCache, OffloadCache

    store = module._parameters
    if not isinstance(store, OffloadCache):
        tensor = store.get("weight")
        return {"tier": "resident", "device": None if tensor is None else str(tensor.device)}
    tensor = store.offloaded_values.get("weight")
    if tensor is None:
        return {"tier": "absent"}
    if tensor.device.type != "meta":
        return {"tier": "cpu", "device": str(tensor.device)}
    if isinstance(store, DiskCache) and tensor in store.index:
        info = store.index[tensor]
        path = Path(info["safetensors_file"])
        return {"tier": "disk", "file": path.name, "weight_name": info["weight_name"],
                "symlink_into_checkpoint": path.is_symlink(), "staged": not path.is_symlink()}
    return {"tier": "meta-unindexed"}


def _offloaded_weight_sha(module) -> str | None:
    """Hash a weight as it sits in the offload store, without onloading it to
    the accelerator. A CPU-tier tensor is hashed in place; a disk-tier tensor
    is read from the file its cache's index names, staged or symlinked, on the
    CPU. Returns None only when there is no weight to hash, so a control
    comparing two of these cannot report "unchanged" for a tier it cannot read
    (the first disk-tier run did exactly that: None == None).
    """
    from compressed_tensors.offload.cache import DiskCache, OffloadCache
    from safetensors import safe_open

    store = module._parameters
    tensor = (store.offloaded_values.get("weight") if isinstance(store, OffloadCache)
              else store.get("weight"))
    if tensor is None:
        return None
    if tensor.device.type != "meta":
        return tensor_sha(tensor.detach())
    if isinstance(store, DiskCache) and tensor in store.index:
        info = store.index[tensor]
        with safe_open(info["safetensors_file"], framework="pt", device="cpu") as handle:
            return tensor_sha(handle.get_tensor(info["weight_name"]))
    raise RuntimeError("weight is a meta tensor the offload store does not index; "
                       "the modifier-entered control cannot read it")


def run_step(model, bundle: Path, manifest: dict, row_ids: list[str], policy: str,
             propagate_error: bool, abort_after: int | None,
             offload_dir: Path, config=None, recipe=None,
             scales_out: Path | None = None) -> dict:
    """One population size through the real pipeline. Returns its measurements."""
    loader = build_loader(bundle, manifest, row_ids)
    tokens = sum(
        next(r for r in manifest["rows"] if r["row_id"] == row_id)["sequence_length"]
        for row_id in row_ids
    )
    visual = sum(
        next(r for r in manifest["rows"] if r["row_id"] == row_id)["vision_positions"]
        for row_id in row_ids
    )
    dataset_args = DatasetArguments(
        dataset=loader,
        sequential_targets=SEQUENTIAL_TARGETS,
        propagate_error=propagate_error,
    )

    longest = max(
        (next(r for r in manifest["rows"] if r["row_id"] == row_id)["sequence_length"]
         for row_id in row_ids), default=0
    )
    records = [next(r for r in manifest["rows"] if r["row_id"] == row_id)
               for row_id in row_ids]
    # Before `cuda_reset`, so the probe's own small allocations cannot enter
    # the step's peak.
    availability = (sdpa_availability_for_rows(records, config, compute_dtype(policy))
                    if config is not None else None)
    cuda_reset()
    peak = _CumulativePeak()
    host_before = host_peak_kib()
    disk_before = directory_bytes(offload_dir)["physical_bytes"]
    started = time.time()
    instrumentation = _Instrumentation()
    instrumentation.peak = peak
    instrumentation.abort_after = abort_after
    outcome, error = "completed", None

    modifier_record: dict | None = None
    control_layer = model.model.language_model.layers[0].self_attn.q_proj
    weight_before = _offloaded_weight_sha(control_layer)
    location_before = _offloaded_weight_location(control_layer)
    recipe_yaml = None
    cwd_before = sorted(str(p) for p in Path.cwd().iterdir())
    with instrumentation.install():
        try:
            with create_session() as session:
                from llmcompressor.modeling.offset_norm import norm_calibration_context

                # `oneshot` wraps calibration in this; Qwen3-VL's RMSNorm is a
                # standard norm so it converts nothing, and it is entered here
                # so the pilot runs the same context a real run would.
                with contextlib.ExitStack() as stack:
                    stack.enter_context(norm_calibration_context(model))
                    session.initialize(model=model, start=-1, recipe=recipe,
                                       calib_data=loader,
                                       sequential_targets=SEQUENTIAL_TARGETS)
                    if recipe is not None:
                        from h3_awq_recipe import assert_decoder_only_boundary

                        modifiers = session.lifecycle.recipe.modifiers
                        # The recipe as the session holds it, serialized while
                        # the session is live: the save wrapper writes
                        # `recipe.yaml` from the *active* session, and the
                        # emit runs after this context has closed, which left
                        # the first candidate's recipe file empty.
                        recipe_yaml = session.lifecycle.recipe.yaml()
                        awq = next(m for m in modifiers if type(m).__name__ == "AWQModifier")
                        # Asserted after the session applied the config and
                        # before any forward: the boundary is what the
                        # modifier will actually rewrite, read off the model.
                        modifier_record = {
                            "modifiers": [type(m).__name__ for m in modifiers],
                            "boundary": assert_decoder_only_boundary(model),
                            "awq_offload_device": str(awq.offload_device),
                            "awq_duo_scaling": awq.duo_scaling,
                            "awq_n_grid": awq.n_grid,
                        }
                        stack.enter_context(instrumentation.instrument_awq(awq))
                    sequential_pipeline.SequentialPipeline()(model, loader, dataset_args)
                session.finalize()
        except DeliberateAbort as exc:
            outcome, error = "deliberate_abort", str(exc)
        except torch.OutOfMemoryError as exc:
            outcome = "cuda_oom"
            error = str(exc).strip().splitlines()[0][:200]
        except Exception as exc:  # a real failure is a result, not a crash
            import traceback
            outcome = "raised"
            error = f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1][:200]}"
            measurement_traceback = traceback.format_exc()
            print(measurement_traceback[-2500:])

    # Unconditionally, after the attempt. An OOM during cache construction
    # happens before the cache wrapper returns and before any subgraph forward,
    # so neither of the observation points inside the instrumentation is ever
    # reached and that peak would otherwise be dropped entirely.
    peak.observe()
    elapsed = time.time() - started
    measurement = {
        "rows": len(row_ids),
        "row_ids": list(row_ids),
        "sequence_tokens": tokens,
        "longest_row_tokens": longest,
        "visual_tokens": visual,
        "rows_detail": [
            {
                "row_id": row_id,
                "sequence_tokens": record["sequence_length"],
                "visual_tokens": record["vision_positions"],
                "vision_blocks": len(record["vision_blocks"]),
                "largest_block_patches": max(
                    (int(b["grid_thw"][0][1]) * int(b["grid_thw"][0][2])
                     for b in record["vision_blocks"]), default=0
                ),
                "nominal_attention_logit_footprints": (
                    nominal_attention_logit_footprints(
                        record, config, compute_dtype(policy)
                    ) if config is not None else None
                ),
            }
            for row_id, record in (
                (r, next(x for x in manifest["rows"] if x["row_id"] == r))
                for r in row_ids
            )
        ],
        "note_on_attribution": "total tokens govern the host-side cache; GPU "
                               "peak is governed by the longest single row and "
                               "its largest vision block, because rows are "
                               "processed one at a time",
        "precision_policy": policy,
        "propagate_error": propagate_error,
        "outcome": outcome,
        "error": error,
        "seconds_total": round(elapsed, 1),
        "seconds_trace": round(instrumentation.trace_seconds, 1),
        "seconds_cache_build": round(instrumentation.cache_build_seconds, 1),
        "seconds_in_subgraph_forwards": round(instrumentation.forward_seconds, 1),
        "subgraph_forward_calls": instrumentation.forward_calls,
        "trace": instrumentation.trace,
        "effective_input": effective_input_record(
            loader.dataset.transforms, row_ids, instrumentation
        ),
        "peak_cuda": peak.as_dict(),
        "host_memory": host_memory(),
        "peak_host_rss_gib": round(host_peak_kib() / 2**20, 2),
        "host_rss_growth_gib": round((host_peak_kib() - host_before) / 2**20, 2),
        "offload_dir_physical_bytes_before": disk_before,
        "offload_dir": directory_bytes(offload_dir),
    }
    peaks = list(instrumentation.subgraph_peak.values())
    if peaks:
        warm = [p for p in peaks if not p["cold_call"]]
        measurement["per_subgraph"] = {
            "calls": len(peaks),
            "cold_calls": sum(1 for p in peaks if p["cold_call"]),
            "max_allocated_before_gib": max(p["allocated_before_gib"] for p in peaks),
            "max_transient_gib": max(p["transient_gib"] for p in peaks),
            "warm_max_allocated_before_gib": (
                max(p["allocated_before_gib"] for p in warm) if warm else None
            ),
            "warm_max_transient_gib": (
                max(p["transient_gib"] for p in warm) if warm else None
            ),
            "note": "a subgraph's weights are onloaded lazily inside its first "
                    "forward, so only the warmed figures can be read as weight "
                    "residency; the cold ones understate it",
        }
    if availability is not None:
        measurement["sdpa_availability"] = availability
    if recipe is not None:
        weight_after = _offloaded_weight_sha(control_layer)
        cwd_after = sorted(str(p) for p in Path.cwd().iterdir())
        modifier_record = modifier_record or {"boundary": None}
        modifier_record.update({
            "recipe_yaml": recipe_yaml if recipe is not None else None,
            "smoothing_calls": instrumentation.awq_smoothing_calls,
            "seconds_in_awq_smoothing": round(instrumentation.awq_smoothing_seconds, 1),
            "parent_reruns": instrumentation.awq_parent_runs,
            "mappings_scaled": len(instrumentation.awq_scales),
            "scales": instrumentation.awq_scales,
            "modifier_entered_control": {
                "module": "language_model.layers.0.self_attn.q_proj",
                "location_before": location_before,
                "location_after": _offloaded_weight_location(control_layer),
                "weight_sha256_before": weight_before,
                "weight_sha256_after": weight_after,
                "weight_changed_in_offload_store": (
                    weight_before is not None and weight_before != weight_after
                ),
                "note": "read from the offload store without onloading; AWQ "
                        "smoothing rewrites the stored weight, so a run that "
                        "entered the modifier path changes this hash",
            },
            "no_files_written": cwd_before == cwd_after,
            "scales_file": None,
        })
        if scales_out is not None and instrumentation.awq_scale_tensors:
            save_file({k: v for k, v in instrumentation.awq_scale_tensors.items()},
                      str(scales_out))
            modifier_record["scales_file"] = scales_out.name
        measurement["modifier"] = modifier_record
    if instrumentation.oom_message:
        measurement["oom"] = {
            "allocator_message": instrumentation.oom_message,
            "stage": instrumentation.oom_stage,
            "on_cold_call": instrumentation.oom_on_cold_call,
            "note": "the allocator's own message, kept because the pipeline's "
                    "wrapper replaces it and a failed allocation never appears "
                    "in max_memory_allocated",
            "compare_against": "rows_detail[].nominal_attention_logit_footprints "
                               "and sdpa_availability, both in this step",
        }
    if instrumentation.cache is not None:
        residual = cache_footprint(instrumentation.cache)
        peak_bytes = instrumentation.cache_peak_bytes
        measurement["intermediates_cache"] = {
            "declared_offload_device": residual["declared_offload_device"],
            "initial": {
                "note": "as materialised by from_dataloader, before any subgraph",
                "tensors": (instrumentation.cache_initial or {}).get("tensors"),
                "gib_by_device": (instrumentation.cache_initial or {}).get("gib_by_device"),
                "keys": (instrumentation.cache_initial or {}).get("keys"),
            },
            "peak": {
                "note": "maximum over samples taken after cache construction "
                        "and after every subgraph forward; this is the growth "
                        "figure, the residual below is not",
                "tensors": instrumentation.cache_peak_tensors,
                "bytes_by_device": peak_bytes,
                "gib_by_device": {k: round(v / 2**30, 3) for k, v in peak_bytes.items()},
                "bytes_per_token": (round(sum(peak_bytes.values()) / tokens, 1)
                                    if tokens else None),
            },
            "residual_after_run": {
                "note": "what the cache still held when the pipeline returned: "
                        "the last subgraph's inputs, because consumed names are "
                        "deleted as the run proceeds",
                "tensors": residual["tensors"],
                "gib_by_device": residual["gib_by_device"],
                "keys": residual["keys"],
            },
        }

    # Cleanup state after the step, whichever way it ended. A run that leaves
    # the accelerator full is a different problem from one that OOMs.
    del loader
    instrumentation.cache = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        measurement["cuda_allocated_after_cleanup_gib"] = round(
            torch.cuda.memory_allocated() / 2**30, 3
        )
        measurement["cuda_reserved_after_cleanup_gib"] = round(
            torch.cuda.memory_reserved() / 2**30, 3
        )
    return measurement


def emit_candidate(model, candidate_dir: Path, source: Path, report: dict) -> dict:
    """The launch: save the compressed candidate, and only the candidate.

    New directory, refused if it exists, refused under the source checkpoint or
    the deployed artifact's config. `save_compressed=True` is what packs the
    W4 weights; the release processor, video processor and tokenizer files are
    copied beside the weights so the candidate declares the release bounds it
    was calibrated at rather than inheriting anything from the deployed
    artifact's constrained snapshot. The pilot report is written beside it as
    the run record.
    """
    from llmcompressor.transformers.compression.compressed_tensors_utils import (
        modify_save_pretrained,
    )

    # `oneshot` installs this wrapper when it loads the model; the pilot loads
    # manually, so it is installed here. Without it the plain transformers
    # `save_pretrained` writes the weights as they are and no
    # `quantization_config`, which the first emit test produced: a 6 GiB
    # two-layer "candidate" that was not quantized at all.
    modify_save_pretrained(model)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(candidate_dir), save_compressed=True)
    # The wrapper's recipe.yaml is written from the active session, which has
    # closed by now; write the recipe captured while it was live.
    recipe_yaml = None
    for step in report.get("steps", []):
        recipe_yaml = (step.get("modifier") or {}).get("recipe_yaml") or recipe_yaml
    if recipe_yaml:
        (candidate_dir / "recipe.yaml").write_text(recipe_yaml)
    copied = []
    for name in ("preprocessor_config.json", "video_preprocessor_config.json",
                 "tokenizer_config.json", "tokenizer.json", "vocab.json",
                 "merges.txt", "chat_template.json"):
        path = source / name
        if path.exists():
            shutil.copyfile(path, candidate_dir / name)
            copied.append(name)
    (candidate_dir / "h3_v2_run_record.json").write_text(json.dumps(report, indent=2) + "\n")
    files = [p for p in candidate_dir.rglob("*") if p.is_file()]
    return {
        "emitted": True,
        "logical_name": candidate_dir.name,
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "processor_files_copied_from_source": copied,
        "note": "a new directory; the deployed artifact, its symlink and the "
                "source checkpoint were not touched. Not deployed by this step.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--row", action="append",
                        help="escalation order; repeatable. Defaults to the "
                             "bundle's declared order")
    parser.add_argument("--source-dir", default=os.environ.get("H3_BF16_ENCODER_DIR"))
    parser.add_argument(
        "--policy", default="comfy_exact", choices=sorted(POLICIES),
        help="defaults to the Gate 1B-accepted policy. A Gate 2 number measured "
             "under a rejected policy is not actionable",
    )
    parser.add_argument("--attention", default="grouped_query", choices=ATTENTION_KINDS,
                        help="which KV form reaches SDPA; the kernel axis. See "
                             "bench/h3_attention_kernel.py")
    parser.add_argument("--layers", type=int, default=50,
                        help="decoder layers to build; 50 is the H3 window")
    parser.add_argument(
        "--offload", default="host", choices=("host", "auto_offload"),
        help="`host` keeps every parameter in host RAM; `auto_offload` uses "
             "llmcompressor's load_context bridge, which converts Accelerate's "
             "hooks into compressed-tensors offload and can spill to disk",
    )
    parser.add_argument("--gpu-gib", type=float, default=14.0,
                        help="recorded, not enforced: the sequential pipeline "
                             "owns onload/offload, see load_model")
    parser.add_argument("--no-propagate-error", action="store_true",
                        help="measure the single-pass cost; the default replays")
    parser.add_argument(
        "--prefix", action="append", type=int,
        help="population sizes to measure; repeatable. Defaults to 1, 3 and all",
    )
    parser.add_argument("--abort-after", type=int,
                        help="control: raise inside the Nth subgraph forward")
    parser.add_argument("--modifier", default="none", choices=("none", "awq"),
                        help="Gate 2B: instantiate the real v2 recipe from "
                             "bench/h3_awq_recipe.py and run with it. One "
                             "prefix per process; nothing is saved")
    parser.add_argument("--awq-duo-scaling", default="false",
                        choices=("false", "true", "both"))
    parser.add_argument("--awq-n-grid", type=int, default=20)
    parser.add_argument("--emit-candidate", default=None, metavar="DIR",
                        help="the launch: after the modifier-bearing run, save the "
                             "compressed candidate to this NEW directory with the "
                             "release processor/tokenizer files beside it. Refused "
                             "if the directory exists or resolves anywhere under "
                             "the deployed artifact or the source checkpoint")
    parser.add_argument("--host-reserve-gib", type=float, default=None,
                        help="explicit host-memory reserve passed to the bridge "
                             "instead of its 5 GB default; recorded either way")
    parser.add_argument("--out", default=str(REPORT))
    args = parser.parse_args()
    if not args.source_dir:
        raise SystemExit("--source-dir or H3_BF16_ENCODER_DIR is required")

    bundle = Path(args.bundle).expanduser().resolve()
    manifest = json.loads((bundle / "presentation.json").read_text())
    order = args.row or manifest["order"]
    known = {r["row_id"] for r in manifest["rows"]}
    unknown = [row for row in order if row not in known]
    if unknown:
        raise SystemExit(f"rows not in the bundle: {unknown}")

    source = Path(args.source_dir).expanduser().resolve()
    offload_dir = Path(tempfile.mkdtemp(prefix="h3-pilot-offload-"))
    print(f"policy {args.policy}: {POLICY_INTENT[args.policy]}")
    print(f"loading {args.layers} decoder layers stored at "
          f"{storage_dtype(args.policy)}, computing at {compute_dtype(args.policy)}; "
          f"offload staging in a temporary directory")

    # Host memory before the load is what the bridge's `max_memory` is derived
    # from (`psutil.virtual_memory().available - extra_cpu_mem`), so the
    # CPU/disk split it chose can only be read against this figure.
    host_before_load = host_memory()
    recipe = None
    recipe_description = None
    candidate_dir = None
    if args.emit_candidate:
        if args.modifier != "awq":
            raise SystemExit("--emit-candidate needs --modifier awq")
        candidate_dir = Path(args.emit_candidate).expanduser().resolve()
        if candidate_dir.exists():
            raise SystemExit(f"refuse to write into an existing directory: {candidate_dir.name}")
        forbidden = [Path(args.source_dir).expanduser().resolve(),
                     (BENCH.parent / "config").resolve()]
        for root in forbidden:
            if root == candidate_dir or root in candidate_dir.parents:
                raise SystemExit("the candidate directory resolves under the source "
                                 "checkpoint or the deployed artifact's config; refused")
    if args.modifier == "awq":
        from h3_awq_recipe import build_recipe, describe_recipe

        if args.offload != "auto_offload":
            raise SystemExit("Gate 2B runs through the converted offload bridge only")
        if len(set(min(max(1, n), len(order)) for n in (args.prefix or [1]))) != 1:
            raise SystemExit("--modifier awq takes exactly one --prefix: AWQ mutates "
                             "the weights, so a second prefix in the same process "
                             "would calibrate on already-smoothed ones")
        duo = {"false": False, "true": True, "both": "both"}[args.awq_duo_scaling]
        # Constructed before the model loads, so a recipe defect fails here and
        # costs nothing.
        recipe = build_recipe(offload_device="cpu", duo_scaling=duo, n_grid=args.awq_n_grid)
        recipe_description = describe_recipe(recipe)
        print(f"recipe: {[type(m).__name__ for m in recipe]}")

    started = time.time()
    model = load_model(source, args.policy, args.layers, args.gpu_gib, offload_dir,
                       args.offload, args.host_reserve_gib)
    load_seconds = time.time() - started
    host_after_load = host_memory()
    staging_after_load = directory_bytes(offload_dir)
    topology = offload_topology(model)
    model_config = model.config
    print(f"  loaded in {load_seconds:.1f}s, {topology['summary']}")
    print(f"  host before {host_before_load} after {host_after_load}")
    print(f"  staged {staging_after_load['physical_bytes'] / 2**30:.1f} GiB in "
          f"{staging_after_load['files']} files")

    report: dict = {
        "pilot": "Gate 2A: llm-compressor SequentialPipeline over native-H3 "
                 "batches, no modifiers, no artifact",
        "gate": "2A",
        "sets_population_budget": False,
        "budget_requires": "Gate 2B, a bounded disposable modifier-bearing run "
                           "that measures the AWQ increment",
        "boundary": "no recipe instantiated, no quantization, no candidate "
                    "directory, nothing saved. Measures the floor beneath a "
                    "calibration run, not the whole of one",
        "path_policy": "logical identifiers only",
        "producer": producer_provenance(__file__),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "llmcompressor": llmcompressor.__version__,
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "gpu_total_gib": (round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
                              if torch.cuda.is_available() else None),
        },
        "precision_policy": {"policy": args.policy, "intent": POLICY_INTENT[args.policy]},
        "model": {
            "logical_name": source.name,
            "decoder_layers_built": args.layers,
            "offload_topology": topology,
            "load_seconds": round(load_seconds, 1),
            "host_memory_before_load": host_before_load,
            "host_memory_after_load": host_after_load,
            "staging_after_load": staging_after_load,
            "gpu_weight_budget_gib_recorded_not_enforced": args.gpu_gib,
            "offload_arrangement": args.offload,
            "bridge_extra_cpu_mem_default_bytes": 5e9,
            "bridge_extra_cpu_mem_used_bytes": (
                5e9 if args.host_reserve_gib is None else int(args.host_reserve_gib * 2**30)
            ),
            "host_reserve_declared": args.host_reserve_gib is not None,
            "bridge_reserve_note": "compressed_tensors load_offloaded_model "
                                   "reserves this much host memory for "
                                   "non-loading work; Gate 2B needs an explicit "
                                   "larger reserve and a fresh model per "
                                   "modifier arm, because AWQ mutates weights "
                                   "and attaches state",
            "device_map_note": "raw device_map='auto' is incompatible with "
                               "SequentialPipeline in this pinned pair "
                               "(accelerate's functools.partial forward vs "
                               "compressed_tensors reading forward.__func__). "
                               "`auto_offload` goes through load_context, which "
                               "converts those hooks, and is a different claim",
        },
        "bundle_provenance": manifest["provenance"],
        "modifier": {
            "kind": args.modifier,
            "recipe": recipe_description if recipe is not None else None,
            "note": ("the real v2 recipe, run without any save; the increment "
                     "over the no-modifier floor is the measurement"
                     if recipe is not None else
                     "no modifier: the floor beneath a calibration run"),
        },
        "steps": [],
    }
    if recipe is not None:
        report["pilot"] = ("Gate 2B: llm-compressor SequentialPipeline over native-H3 "
                           "batches with the real AWQ recipe, no artifact")
        report["gate"] = "2B"

    try:
        with calibration_precision(model, args.policy) as precision, \
                attention_kernel(model, args.attention) as kernel:
            report["precision_policy"].update(precision)
            report["attention_kernel"] = kernel
            sizes = sorted({min(max(1, n), len(order))
                            for n in (args.prefix or [1, 3, len(order)])})
            for size in sizes:
                rows = order[:size]
                print(f"\nstep {size}: {rows[-1]}")
                out_path = Path(args.out).expanduser().resolve()
                measurement = run_step(
                    model, bundle, manifest, rows, args.policy,
                    not args.no_propagate_error,
                    args.abort_after if size == sizes[-1] else None,
                    offload_dir, model_config, recipe,
                    out_path.with_name(out_path.stem + "_awq_scales.safetensors")
                    if recipe is not None else None,
                )
                if measurement.get("modifier"):
                    mod = measurement["modifier"]
                    print(f"  awq smoothing calls {mod['smoothing_calls']}  parent reruns "
                          f"{mod['parent_reruns']}  mappings {mod['mappings_scaled']}  "
                          f"{mod['seconds_in_awq_smoothing']}s  weight changed "
                          f"{mod['modifier_entered_control']['weight_changed_in_offload_store']}")
                report["steps"].append(measurement)
                print(f"  {measurement['outcome']}  tokens {measurement['sequence_tokens']}  "
                      f"cuda {measurement['peak_cuda']}  host peak "
                      f"{measurement['peak_host_rss_gib']} GiB  "
                      f"forwards {measurement['subgraph_forward_calls']}  "
                      f"{measurement['seconds_total']}s")
                if measurement.get("intermediates_cache"):
                    cache = measurement["intermediates_cache"]
                    print(f"  cache peak {cache['peak']['gib_by_device']}  "
                          f"residual {cache['residual_after_run']['gib_by_device']}")
                if measurement["outcome"] not in ("completed", "deliberate_abort"):
                    print(f"  stopping the escalation here: {measurement['error']}")
                    break
        if candidate_dir is not None and report["steps"] \
                and report["steps"][-1]["outcome"] == "completed" \
                and report["steps"][-1].get("modifier", {}).get(
                    "modifier_entered_control", {}).get("weight_changed_in_offload_store"):
            report["candidate"] = emit_candidate(model, candidate_dir, source, report)
            print(f"candidate emitted to {candidate_dir.name}: "
                  f"{report['candidate']['bytes'] / 2**30:.2f} GiB in "
                  f"{report['candidate']['files']} files")
            candidate_dir = None
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        report["offload_dir_at_exit"] = directory_bytes(offload_dir)
        shutil.rmtree(offload_dir, ignore_errors=True)

    if candidate_dir is not None:
        report["candidate"] = {"emitted": False,
                               "reason": "the modifier-bearing step did not complete "
                                         "with a changed weight; nothing was saved"}
        print("candidate NOT emitted: the run did not complete cleanly")
    completed = [s for s in report["steps"] if s["outcome"] == "completed"]
    report["sequential_floor_envelope"] = {
        "largest_completed_sequence_tokens": (
            max((s["sequence_tokens"] for s in completed), default=0)
        ),
        "largest_completed_rows": max((s["rows"] for s in completed), default=0),
        "is_a_budget": False,
        "note": "a token total this MODIFIER-FREE path completed on this box. "
                "It is the floor beneath a calibration run: tracing, the "
                "intermediates cache and two forwards per subgraph per row, "
                "with no AWQ modifier observing or rewriting anything. It is "
                "not a chosen population, not a manifest, and it cannot set "
                "the final budget, which needs a modifier-bearing pass",
    }
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote results/{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
