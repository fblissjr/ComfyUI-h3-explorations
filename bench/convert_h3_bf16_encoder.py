"""Adapt a full-depth HF Qwen3-VL-32B text encoder to the ComfyUI H3 encoder.

The MiniMax H3 release ships the complete Qwen3-VL-32B (64 decoder layers,
`lm_head`, final norm).  Native ComfyUI consumes far less than that, and the
part it consumes is not a judgement call -- it is fixed in
`comfy/text_encoders/llama.py::Qwen3VL_32BConfig`:

    num_hidden_layers: int = 50
    lm_head: bool = False
    final_norm: bool = False

Those are hardcoded, not read from the checkpoint, so layers 50-63, `lm_head`
and `model.language_model.norm` are never instantiated.  Shipping them costs
file size and load bandwidth and buys nothing.

The rename is the half that is easy to miss and is NOT optional.  Core
recognises this encoder in `comfy/sd.py::detect_te_model` by

    "visual.deepstack_merger_list.0.norm.weight" in sd
    and "model.layers.49.self_attn.q_proj.weight" in sd

and the QWEN3VL_32B branch in `load_text_encoder_state_dicts` is the one
qwen3vl branch that does NOT apply `state_dict_prefix_replace`.  A checkpoint
left in HF naming (`model.language_model.`, `model.visual.`) therefore matches
the *earlier* `model.visual.deepstack_merger_list.0.norm.weight` test first and
is misdetected as QWEN3VL_8B, which then fails on a width mismatch.  That is
the 2026-08-23 escape recorded in `h3_awq_encoder.py::load_h3_awq_encoder`.

This is the bf16 counterpart of `h3_awq_encoder.adapt_compressed_state_dict`,
which does the same selection and rename for the W4A16 lane.  The drop rule is
deliberately shared with it via `_drop_source_key`, so the two lanes cannot
disagree about what H3 consumes.

No tensor is decoded: dtype and bytes are copied verbatim from the source, so
this is a subset-and-rename of the container and the surviving weights are
bit-identical to the release.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parents[1]
sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(REPO))

from h3_awq_encoder import H3_LAYERS, _drop_source_key  # noqa: E402

# Matches the `minimax_h3_te` key the shipped Comfy-Org INT8 artifact carries.
METADATA = {
    "minimax_h3_te": json.dumps(
        {"num_hidden_layers": H3_LAYERS, "output": "unnormalized_hidden_after_layer_50"},
        separators=(", ", ": "),
    )
}

_RENAMES = (("model.language_model.", "model."), ("model.visual.", "visual."))
_CHUNK = 64 << 20


def read_header(path: Path) -> tuple[dict, int]:
    """Return (header, byte offset where the tensor data buffer starts)."""
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    return header, 8 + size


def target_name(name: str) -> str:
    for prefix, replacement in _RENAMES:
        if name.startswith(prefix):
            return replacement + name[len(prefix):]
    return name


def plan(header: dict, depth: int) -> tuple[list[tuple[str, str, dict]], int, int]:
    """Select and rename surviving tensors, preserving source byte order.

    Source order is kept so the copy reads the input strictly forwards.
    """
    kept, dropped_bytes = [], 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        if _drop_source_key(name, depth):
            dropped_bytes += end - start
            continue
        kept.append((name, target_name(name), entry))
    kept.sort(key=lambda item: item[2]["data_offsets"][0])
    kept_bytes = sum(e["data_offsets"][1] - e["data_offsets"][0] for _, _, e in kept)
    return kept, kept_bytes, dropped_bytes


def convert(src: Path, dst: Path, depth: int) -> dict:
    header, data_start = read_header(src)
    kept, kept_bytes, dropped_bytes = plan(header, depth)
    if not kept:
        raise ValueError(f"{src.name}: nothing survived the H3 selection")

    out_header: dict = {"__metadata__": METADATA}
    cursor = 0
    for _, new_name, entry in kept:
        length = entry["data_offsets"][1] - entry["data_offsets"][0]
        out_header[new_name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [cursor, cursor + length],
        }
        cursor += length

    blob = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)  # data buffer stays 8-byte aligned

    tmp = dst.with_suffix(dst.suffix + ".partial")
    with src.open("rb") as fin, tmp.open("wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        for _, _, entry in kept:
            start, end = entry["data_offsets"]
            fin.seek(data_start + start)
            remaining = end - start
            while remaining:
                chunk = fin.read(min(_CHUNK, remaining))
                if not chunk:
                    raise EOFError("source ended inside a tensor; file is truncated")
                fout.write(chunk)
                remaining -= len(chunk)
    tmp.replace(dst)
    return {"kept": len(kept), "kept_bytes": kept_bytes, "dropped_bytes": dropped_bytes}


def verify(dst: Path, control: Path | None) -> None:
    """Grade the result against core's detector and, if given, a control artifact.

    The detector is the thing that actually decides whether this file loads as
    H3, so it is asserted rather than the key names it happens to test.  The
    control is Comfy-Org's own INT8 conversion: an independent implementation
    of this same selection, so a key-set difference against it is a real
    disagreement rather than a restatement of what this script just wrote.
    """
    import comfy.sd

    class Fake:
        def __init__(self, shape):
            self.shape = tuple(shape)

    header, _ = read_header(dst)
    header.pop("__metadata__", None)
    sd = {name: Fake(entry["shape"]) for name, entry in header.items()}

    detected = comfy.sd.detect_te_model(sd)
    if detected != comfy.sd.TEModel.QWEN3VL_32B:
        raise SystemExit(f"FAIL detect_te_model returned {detected}, not QWEN3VL_32B")
    print(f"  detect_te_model -> {detected}")

    if control is None:
        print("  control        -> none given (key set ungraded)")
        return
    ctrl, _ = read_header(control)
    ctrl.pop("__metadata__", None)
    # Strip the control's quantization companions; what remains is the logical
    # tensor set a bf16 artifact must carry.
    logical = {
        re.sub(r"\.(weight_scale|comfy_quant)$", ".weight", name) for name in ctrl
    }
    missing, extra = logical - set(sd), set(sd) - logical
    for label, names in (("missing vs control", missing), ("extra vs control", extra)):
        if names:
            shown = ", ".join(sorted(names)[:5])
            raise SystemExit(f"FAIL {label} ({len(names)}): {shown}")
    print(f"  control        -> key set matches {control.name} ({len(logical)} tensors)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path, help="full-depth HF-named bf16 safetensors")
    ap.add_argument("dst", type=Path, help="ComfyUI-named H3 encoder to write")
    ap.add_argument("--depth", type=int, default=H3_LAYERS,
                    help=f"decoder layers to keep (default {H3_LAYERS}, core's fixed count)")
    ap.add_argument("--control", type=Path, default=None,
                    help="an existing working H3 encoder to grade the key set against")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    args = ap.parse_args()

    header, _ = read_header(args.src)
    kept, kept_bytes, dropped_bytes = plan(header, args.depth)
    total = kept_bytes + dropped_bytes
    print(f"{args.src.name}")
    print(f"  source  {total / 1e9:8.2f} GB  {len(header) - 1:5d} tensors")
    print(f"  keep    {kept_bytes / 1e9:8.2f} GB  {len(kept):5d} tensors")
    print(f"  drop    {dropped_bytes / 1e9:8.2f} GB  "
          f"{len(header) - 1 - len(kept):5d} tensors  ({100 * dropped_bytes / total:.1f}%)")
    renamed = sum(1 for old, new, _ in kept if old != new)
    print(f"  rename  {renamed:5d} tensors into ComfyUI H3 naming")
    if args.dry_run:
        print("  (dry run, nothing written)")
        return 0

    stats = convert(args.src, args.dst, args.depth)
    print(f"  wrote   {args.dst}  ({stats['kept_bytes'] / 1e9:.2f} GB)")
    verify(args.dst, args.control)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
