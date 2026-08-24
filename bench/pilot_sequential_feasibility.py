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
nothing to emit: no recipe is instantiated, no save is called, no output
directory is created.

It answers, by measuring:

- peak allocated and reserved VRAM, per stage;
- peak host RAM, as the kernel's own high-water mark;
- where the intermediates cache lives and how it grows with tokens;
- whether the replay pass really runs, counted at `Subgraph.forward`;
- time by observable stage;
- temporary disk use under the offload directory; and
- what is left behind after a deliberate abort.

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

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
REPORT = BENCH / "results" / "2026-08-24_sequential_feasibility_pilot.json"

from h3_calibration_precision import (  # noqa: E402
    POLICIES,
    POLICY_INTENT,
    calibration_precision,
    compute_dtype,
)
from h3_effective_batch import effective_batch  # noqa: E402

SEQUENTIAL_TARGETS = ["Qwen3VLTextDecoderLayer"]


# --------------------------------------------------------------------------
# measurement


def host_peak_kib() -> int:
    """The kernel's own high-water mark for this process, not a sampled guess."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


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


def directory_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            with contextlib.suppress(OSError):
                total += entry.stat().st_size
    return total


def cache_footprint(cache) -> dict:
    """Where the intermediates cache actually lives, and how large it is.

    Read off the cache's own structure rather than inferred from row count:
    `IntermediatesCache` stores an `IntermediateValue` per key per batch, and
    the offload device is a property of how it was constructed, so a run that
    silently kept activations on the accelerator would show up here.
    """
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
    return {
        "tensors": tensors,
        "bytes_by_device": by_device,
        "gib_by_device": {k: round(v / 2**30, 3) for k, v in by_device.items()},
        "declared_offload_device": str(cache.offload_device),
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
        self.abort_after: int | None = None

    @contextlib.contextmanager
    def install(self):
        original_forward = Subgraph.forward
        original_cache = sequential_pipeline.IntermediatesCache.from_dataloader
        instrumentation = self

        def forward(self, *args, **kwargs):
            if (instrumentation.abort_after is not None
                    and instrumentation.forward_calls >= instrumentation.abort_after):
                raise DeliberateAbort(
                    f"aborting after {instrumentation.forward_calls} subgraph forwards"
                )
            started = time.time()
            try:
                return original_forward(self, *args, **kwargs)
            finally:
                instrumentation.forward_calls += 1
                instrumentation.forward_seconds += time.time() - started
                instrumentation.subgraph_peak[instrumentation.forward_calls] = cuda_peak()

        # `original_cache` is already a bound classmethod, and after the
        # library's own decoration it can be a `functools.partial` with no
        # `__func__`. Calling the bound object is the form that works whatever
        # it is wrapped in; reaching for `__func__` broke on the first real run.
        def from_dataloader(dataloader, model_device, offload_device):
            started = time.time()
            cache = original_cache(dataloader, model_device, offload_device)
            instrumentation.cache_build_seconds = time.time() - started
            instrumentation.cache = cache
            return cache

        Subgraph.forward = forward
        sequential_pipeline.IntermediatesCache.from_dataloader = staticmethod(from_dataloader)
        try:
            yield self
        finally:
            Subgraph.forward = original_forward
            sequential_pipeline.IntermediatesCache.from_dataloader = original_cache


class DeliberateAbort(RuntimeError):
    """The controlled failure. Raised from inside a subgraph forward."""


# --------------------------------------------------------------------------


def load_model(source: Path, policy: str, layers: int, gpu_gib: float,
               offload_dir: Path):
    """Load plainly on the host and let the pipeline own onload/offload.

    **Not `device_map="auto"`.** Accelerate's dispatch replaces every hooked
    module's `forward` with a `functools.partial`, and
    `compressed_tensors.offload.module.offload_module` -- which
    `SequentialPipeline` calls through `set_onload_device` before the first
    batch -- reads `module.forward.__func__`. The two offload mechanisms do not
    compose, and the pipeline dies on the partial before any calibration
    happens. Measured on the first real run of this pilot; it is a property of
    the pinned library pair, not of this harness, and any launcher that reaches
    for `device_map` will meet it.

    So the model is loaded on the host and the sequential pipeline moves each
    subgraph to the accelerator itself, which is the arrangement `oneshot`
    expects. `gpu_gib` is therefore recorded rather than enforced here.
    """
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(source)
    config.text_config.num_hidden_layers = layers
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        source, config=config, dtype=compute_dtype(policy),
        attn_implementation="sdpa",
    ).eval()
    return model


def run_step(model, bundle: Path, manifest: dict, row_ids: list[str], policy: str,
             propagate_error: bool, abort_after: int | None,
             offload_dir: Path) -> dict:
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

    cuda_reset()
    host_before = host_peak_kib()
    disk_before = directory_bytes(offload_dir)
    started = time.time()
    instrumentation = _Instrumentation()
    instrumentation.abort_after = abort_after
    outcome, error = "completed", None

    with instrumentation.install():
        try:
            with create_session() as session:
                session.initialize(model=model, start=-1, recipe=None,
                                   calib_data=loader,
                                   sequential_targets=SEQUENTIAL_TARGETS)
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

    elapsed = time.time() - started
    measurement = {
        "rows": len(row_ids),
        "row_ids": list(row_ids),
        "sequence_tokens": tokens,
        "visual_tokens": visual,
        "precision_policy": policy,
        "propagate_error": propagate_error,
        "outcome": outcome,
        "error": error,
        "seconds_total": round(elapsed, 1),
        "seconds_cache_build": round(instrumentation.cache_build_seconds, 1),
        "seconds_in_subgraph_forwards": round(instrumentation.forward_seconds, 1),
        "subgraph_forward_calls": instrumentation.forward_calls,
        "peak_cuda": cuda_peak(),
        "peak_host_rss_gib": round(host_peak_kib() / 2**20, 2),
        "host_rss_growth_gib": round((host_peak_kib() - host_before) / 2**20, 2),
        "offload_dir_bytes_before": disk_before,
        "offload_dir_bytes_after": directory_bytes(offload_dir),
    }
    if instrumentation.cache is not None:
        measurement["intermediates_cache"] = cache_footprint(instrumentation.cache)
        if tokens:
            total = sum(measurement["intermediates_cache"]["bytes_by_device"].values())
            measurement["intermediates_cache"]["bytes_per_token"] = round(total / tokens, 1)

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
    parser.add_argument("--layers", type=int, default=50,
                        help="decoder layers to build; 50 is the H3 window")
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
    print(f"loading {args.layers} decoder layers at "
          f"{compute_dtype(args.policy)}; offload staging in a temporary directory")

    started = time.time()
    model = load_model(source, args.policy, args.layers, args.gpu_gib, offload_dir)
    load_seconds = time.time() - started
    parameter_dtypes = sorted({str(p.dtype).removeprefix("torch.")
                               for p in model.parameters()})
    devices = sorted({str(p.device) for p in model.parameters()})
    print(f"  loaded in {load_seconds:.1f}s, dtypes {parameter_dtypes}, devices {devices}")

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
            "parameter_dtypes": parameter_dtypes,
            "parameter_devices": devices,
            "load_seconds": round(load_seconds, 1),
            "gpu_weight_budget_gib_recorded_not_enforced": args.gpu_gib,
            "device_map": "none; loaded on the host so compressed_tensors owns "
                          "offload. accelerate's device_map is incompatible "
                          "with SequentialPipeline in this pinned pair",
        },
        "bundle_provenance": manifest["provenance"],
        "steps": [],
    }

    try:
        with calibration_precision(model, args.policy) as precision:
            report["precision_policy"].update(precision)
            sizes = sorted({min(max(1, n), len(order))
                            for n in (args.prefix or [1, 3, len(order)])})
            for size in sizes:
                rows = order[:size]
                print(f"\nstep {size}: {rows[-1]}")
                measurement = run_step(
                    model, bundle, manifest, rows, args.policy,
                    not args.no_propagate_error,
                    args.abort_after if size == sizes[-1] else None,
                    offload_dir,
                )
                report["steps"].append(measurement)
                print(f"  {measurement['outcome']}  tokens {measurement['sequence_tokens']}  "
                      f"cuda {measurement['peak_cuda']}  host peak "
                      f"{measurement['peak_host_rss_gib']} GiB  "
                      f"forwards {measurement['subgraph_forward_calls']}  "
                      f"{measurement['seconds_total']}s")
                if measurement.get("intermediates_cache"):
                    print(f"  cache {measurement['intermediates_cache']['gib_by_device']}")
                if measurement["outcome"] not in ("completed", "deliberate_abort"):
                    print(f"  stopping the escalation here: {measurement['error']}")
                    break
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        report["offload_dir_bytes_at_exit"] = directory_bytes(offload_dir)
        shutil.rmtree(offload_dir, ignore_errors=True)

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
