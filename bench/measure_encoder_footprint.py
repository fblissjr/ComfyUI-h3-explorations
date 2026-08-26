#!/usr/bin/env python3
"""How much VRAM does each H3 conditioning encoder actually need resident?

The four-encoder holdout table
(`bench/results/2026-08-25_four_encoders_holdout_layer50.json`) ranked the
encoder files by fidelity and said nothing about what each one costs to hold.
The cost question was answered by file size, which is the wrong number twice
over: H3 consumes decoder layers 0--49 and never the output head, so a file
carrying all 64 layers plus an `lm_head` overstates its own H3 path, and two
files with the same byte count can split those bytes very differently between
the decoder, the embedding table and the vision tower -- which is what decides
whether there is anything left to compress without touching the vision side.

So this reads the safetensors header of each encoder, buckets every tensor by
the component it belongs to, and reports both the on-disk total and the
**H3 path**: layers 0--49, the embedding table, the vision tower, and whatever
else is neither a later layer nor the head. No tensor data is read, no torch,
no CUDA, either virtualenv.

The layer sum is exact, never a per-layer average extrapolated to fifty: a file
whose layers differ in size (a mixed-precision candidate, which is the point of
measuring this at all) would be misreported by an average. If the layer indices
are not a contiguous run from zero the report says so and derives no H3 path,
because the bucket rule cannot be trusted on a layout it did not expect.

Paths stay out of the record: encoders are named by file name only, resolved
under a models directory given on the command line or `--models-dir`.

    python bench/measure_encoder_footprint.py \\
        --out bench/results/<date>_encoder_footprints.json [--budget-gib 24]

`--budget-gib` is optional and only ever adds a margin column; no card size is
baked in here.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
sys.path.insert(0, str(BENCH))

from h3_producer_provenance import producer_provenance  # noqa: E402

#: Decoder layers H3 consumes, as a count. The tap is layer index 49 and the
#: released depth contract is `canonical/encoder_depth_and_embedding.md`;
#: `comfy/text_encoders/minimax.py` truncates to this many on load.
H3_LAYERS = 50

GIB = float(1 << 30)

#: The unquantized reference, measured alongside the artifacts so the table
#: carries what "no quantization" costs. Named here rather than in
#: `workflows/h3_config.py` because no graph loads it; the file, its byte count
#: and its depth are recorded in
#: `canonical/encoder_depth_and_embedding.md`.
BF16_REFERENCE = "qwen3vl_32b_minimax_h3_bf16.safetensors"


def _bucket(name: str) -> tuple[str, int | None]:
    """Which component a tensor belongs to, and its decoder layer if it has one.

    Ordered so the vision tower wins before the layer rule: the tower has its
    own `.layers.N.` blocks under `visual.`, and counting those as decoder
    layers would both inflate the decoder and invent layer indices the language
    stack does not have.
    """
    if "visual" in name:
        return "vision_tower", None
    if "embed_tokens" in name:
        return "embeddings", None
    if "lm_head" in name:
        return "output_head", None
    if ".layers." in name:
        try:
            index = int(name.split(".layers.", 1)[1].split(".", 1)[0])
        except (IndexError, ValueError):
            return "unparsed_layer", None
        return "decoder_layers", index
    return "other", None


def read_header(path: Path) -> dict:
    """The safetensors header alone: eight bytes of length, then that much JSON."""
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size))


def measure(path: Path) -> dict:
    header = read_header(path)
    buckets: dict[str, int] = {}
    dtypes: dict[str, int] = {}
    per_layer: dict[int, int] = {}
    tensors = 0

    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        nbytes = end - start
        tensors += 1
        dtypes[entry["dtype"]] = dtypes.get(entry["dtype"], 0) + nbytes
        bucket, layer = _bucket(name)
        buckets[bucket] = buckets.get(bucket, 0) + nbytes
        if layer is not None:
            per_layer[layer] = per_layer.get(layer, 0) + nbytes

    record: dict = {
        "file": path.name,
        "tensors": tensors,
        "bytes_total": sum(buckets.values()),
        "bytes_by_component": dict(sorted(buckets.items(),
                                          key=lambda kv: -kv[1])),
        "bytes_by_dtype": dict(sorted(dtypes.items(), key=lambda kv: -kv[1])),
    }

    if not per_layer:
        record["h3_path"] = None
        record["h3_path_refused"] = ("no `.layers.N.` tensors outside the vision "
                                     "tower; this layout is not one the bucket "
                                     "rule was written for")
        return record

    indices = sorted(per_layer)
    record["decoder_layer_indices"] = {"min": indices[0], "max": indices[-1],
                                       "count": len(indices)}
    contiguous = indices == list(range(indices[-1] + 1))
    record["decoder_layers_contiguous_from_zero"] = contiguous
    if not contiguous:
        record["h3_path"] = None
        record["h3_path_refused"] = (
            f"decoder layer indices are not a contiguous run from zero "
            f"({indices[0]}..{indices[-1]}, {len(indices)} present); the H3 "
            f"path cannot be summed without knowing which layers are missing"
        )
        return record
    if len(indices) < H3_LAYERS:
        record["h3_path"] = None
        record["h3_path_refused"] = (
            f"the file carries {len(indices)} decoder layers, fewer than the "
            f"{H3_LAYERS} H3 consumes; it cannot serve as an H3 encoder"
        )
        return record

    kept = sum(per_layer[i] for i in range(H3_LAYERS))
    dropped_layers = sum(per_layer[i] for i in indices[H3_LAYERS:])
    head = buckets.get("output_head", 0)
    resident = record["bytes_total"] - dropped_layers - head
    record["h3_path"] = {
        "layers_kept": H3_LAYERS,
        "layers_dropped": len(indices) - H3_LAYERS,
        "bytes_decoder_kept": kept,
        "bytes_dropped_layers": dropped_layers,
        "bytes_dropped_output_head": head,
        "bytes_resident": resident,
        "gib_resident": round(resident / GIB, 3),
        "note": "weights only: activations, the offload staging copy and the "
                "allocator's overhead are not in this number",
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="what each H3 conditioning encoder costs to hold resident")
    parser.add_argument("--models-dir", default=None,
                        help="directory holding the encoder files; defaults to "
                             "the ComfyUI install's text_encoders beside this repo")
    parser.add_argument("--encoder", action="append", default=None,
                        help="file name to measure; repeatable. Defaults to the "
                             "encoders h3_config names plus the W4A16 pair")
    parser.add_argument("--budget-gib", type=float, default=None,
                        help="optional: report each H3 path against this budget")
    parser.add_argument("--out", default=None, help="write the report here")
    args = parser.parse_args()

    if args.models_dir:
        models = Path(args.models_dir).expanduser().resolve()
    else:
        models = REPO.parents[1] / "models" / "text_encoders"
    if not models.is_dir():
        print(f"no such models directory: {models.name}", file=sys.stderr)
        return 2

    if args.encoder:
        names = list(args.encoder)
    else:
        sys.path.insert(0, str(REPO / "workflows"))
        import h3_config

        names = [BF16_REFERENCE, h3_config.ENCODER_INT8,
                 *sorted(h3_config.CORE_LOADED_ENCODERS - {h3_config.ENCODER_INT8}),
                 h3_config.MODELS["clip"], h3_config.ENCODER_V2]

    rows, missing = [], []
    for name in dict.fromkeys(names):
        path = models / name
        if not path.is_file():
            missing.append(name)
            continue
        rows.append(measure(path))

    report = {
        "question": "what does each H3 conditioning encoder cost to hold "
                    "resident, by component",
        "method": "safetensors header only; tensor bytes bucketed by component; "
                  f"the H3 path is decoder layers 0..{H3_LAYERS - 1} plus the "
                  "embedding table, the vision tower and any remaining tensor, "
                  "with later layers and the output head dropped as "
                  "`comfy/text_encoders/minimax.py` drops them on load",
        "path_policy": "logical identifiers only; encoders named by file name",
        "h3_layers": H3_LAYERS,
        "encoders": rows,
        "encoders_not_found": missing,
        "not_established": "runtime peak, which adds activations, offload "
                           "staging and allocator overhead, and which "
                           "`bench/instrument_render_occupancy.py` measures "
                           "on a real render",
        "producer": producer_provenance(__file__),
    }
    if args.budget_gib is not None:
        report["budget_gib"] = args.budget_gib
        for row in rows:
            path_row = row.get("h3_path")
            if path_row:
                path_row["gib_under_budget"] = round(
                    args.budget_gib - path_row["gib_resident"], 3)

    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.write_text(json.dumps(report, indent=2) + "\n")

    width = max((len(r["file"]) for r in rows), default=0)
    for row in rows:
        path_row = row.get("h3_path")
        total = row["bytes_total"] / GIB
        if path_row is None:
            print(f"{row['file']:<{width}}  {total:6.2f} GiB on disk  "
                  f"H3 path refused: {row['h3_path_refused']}")
            continue
        parts = row["bytes_by_component"]
        margin = path_row.get("gib_under_budget")
        print(f"{row['file']:<{width}}  {total:6.2f} GiB on disk  "
              f"H3 path {path_row['gib_resident']:6.2f} GiB"
              + (f"  ({margin:+.2f} vs budget)" if margin is not None else "")
              + f"  [decoder {path_row['bytes_decoder_kept'] / GIB:.2f}"
                f"  embeddings {parts.get('embeddings', 0) / GIB:.2f}"
                f"  vision {parts.get('vision_tower', 0) / GIB:.2f}]")
    for name in missing:
        print(f"{name}: not present under the models directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
