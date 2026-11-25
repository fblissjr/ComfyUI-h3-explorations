"""Exact overlays of one int8 H3 checkpoint on another, stored per piece.

A research checkpoint built on a released one (FastH3 V2 on fl2va) is the base
plus a few changes, so it can travel as those changes instead of a 22 GB file.
`bench/results/2026-09-29_fasth3_overlay_size.md` measured what the changes
are. An overlay carries three kinds of tensor, each named for the base key it
touches:

- `idx.<key>` and `delta.<key>`: an int8 linear both files hold as int8, as the
  flat row-major positions of the codes that differ and the int8 difference
  target - base at each. The new codes are the base's plus the delta, exactly.
- `full.<key>`: a tensor that is new, differs in dtype or shape, or is small
  enough that a whole copy is cheaper than positions (row scales, norms, adaln).

Every tensor belongs to one piece, so a loader can apply some of them:

    backbone.<n>   block n's linear codes and row scales, and its norms
    gates          every `to_gate_compress` tensor (they exist only in the target)
    refiner        every `token_refiner.` tensor that differs or is new
    adaln          every `adaln_proj` tensor and `adaln_t_table` (they go together)
    io             the tensors outside the blocks: patch and condition
                   projections, the final layer

The file's metadata holds the format, the base file's name and sha256, the
target's name and sha256, and the piece table. The base is checked by content
before anything is applied: an overlay on any other file is refused.

`apply_overlay` never changes the base's tensors: a coded linear is cloned
before its positions are written, and every other tensor is shared. Applying
all pieces gives the target tensor for tensor; `bench/check_checkpoint_overlay.py`
is that proof.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "h3-checkpoint-overlay/1"
GATE_SCALE_SUFFIX = ".attn.to_gate_compress.weight_scale"
INT32_MAX = 2 ** 31 - 1


def piece_of(key: str) -> str:
    if "to_gate_compress" in key:
        return "gates"
    if key.startswith("token_refiner."):
        return "refiner"
    if "adaln_proj" in key or key == "adaln_t_table":
        return "adaln"
    block = re.match(r"blocks\.(\d+)\.", key)
    return f"backbone.{block.group(1)}" if block else "io"


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@functools.lru_cache(maxsize=8)
def _sha256_of(path: str, size: int, mtime_ns: int) -> str:
    # size and mtime_ns are the cache key: a changed file is hashed again
    return file_sha256(path)


def check_base(meta: dict, base_path: str) -> None:
    st = os.stat(base_path)
    got = _sha256_of(base_path, st.st_size, st.st_mtime_ns)
    if got != meta["base_sha256"]:
        raise ValueError(f"this overlay is for {meta['base_name']} (sha256 {meta['base_sha256']}); "
                         f"{base_path} is sha256 {got}")


def build_overlay(base_path: str, target_path: str, out_path: str) -> dict:
    """Write the overlay of `target_path` on `base_path`; return what it holds."""
    tensors, pieces = {}, {}
    stats = {"coded_linears": 0, "coded_positions": 0, "coded_codes": 0, "max_abs_delta": 0}

    def put(name, tensor, piece):
        tensors[name] = tensor.contiguous()
        pieces.setdefault(piece, []).append(name)

    with safe_open(base_path, "pt") as fb, safe_open(target_path, "pt") as ft:
        base_keys, target_keys = set(fb.keys()), set(ft.keys())
        dropped = sorted(base_keys - target_keys)
        if dropped:
            raise ValueError(f"the target lacks {len(dropped)} of the base's tensors, e.g. {dropped[:3]}; "
                             "an overlay adds and replaces, it cannot remove")
        for key in sorted(target_keys):
            new, piece = ft.get_tensor(key), piece_of(key)
            if key not in base_keys:
                put(f"full.{key}", new, piece)
                continue
            old = fb.get_tensor(key)
            if old.dtype == new.dtype and old.shape == new.shape and torch.equal(old, new):
                continue
            if old.dtype == new.dtype == torch.int8 and old.shape == new.shape and key.endswith(".weight") and new.ndim == 2:
                delta = (new.int() - old.int()).flatten()
                idx = delta.nonzero().flatten()
                if idx.numel() * 5 < delta.numel():
                    if delta.numel() > INT32_MAX:
                        raise ValueError(f"{key} has too many codes for int32 positions")
                    values = delta[idx]
                    put(f"idx.{key}", idx.to(torch.int32), piece)
                    put(f"delta.{key}", values.to(torch.int8), piece)
                    stats["coded_linears"] += 1
                    stats["coded_positions"] += idx.numel()
                    stats["coded_codes"] += delta.numel()
                    stats["max_abs_delta"] = max(stats["max_abs_delta"], int(values.abs().max()))
                    continue
            put(f"full.{key}", new, piece)

    metadata = {"format": FORMAT,
                "base_name": os.path.basename(base_path), "base_sha256": file_sha256(base_path),
                "target_name": os.path.basename(target_path), "target_sha256": file_sha256(target_path),
                "pieces": json.dumps(pieces)}
    save_file(tensors, out_path, metadata=metadata)
    bytes_by_piece = {p: sum(tensors[n].numel() * tensors[n].element_size() for n in names)
                      for p, names in sorted(pieces.items())}
    return {**stats, "bytes_by_piece": bytes_by_piece}


def read_overlay(overlay_path: str) -> tuple[dict, dict]:
    """The overlay's metadata and its piece table."""
    with safe_open(overlay_path, "pt") as f:
        meta = f.metadata() or {}
    if meta.get("format") != FORMAT:
        raise ValueError(f"{overlay_path} is not a {FORMAT} file")
    return meta, json.loads(meta["pieces"])


def select_pieces(pieces: dict, blocks, gates: bool, refiner: bool, adaln: bool, io_layers: bool) -> list[str]:
    """The overlay's pieces a selection keeps. A piece the overlay does not carry selects nothing."""
    toggles = {"gates": gates, "refiner": refiner, "adaln": adaln, "io": io_layers}
    return [p for p in pieces
            if (int(p.split(".")[1]) in blocks if p.startswith("backbone.") else toggles[p])]


def apply_overlay(base_sd: dict, overlay_path: str, labels: list[str], gate_scale: float = 1.0) -> dict:
    """`base_sd` with the overlay's pieces `labels` applied, as a new dict.

    `gate_scale` multiplies every `to_gate_compress` row scale the result holds,
    in float32; 1.0 leaves them as stored. A scale with no gates to apply it to
    is an error, not a no-op.
    """
    _, pieces = read_overlay(overlay_path)
    wanted = {name for label in labels for name in pieces[label]}
    out = dict(base_sd)
    with safe_open(overlay_path, "pt") as fo:
        for name in sorted(wanted):
            kind, key = name.split(".", 1)
            if kind == "full":
                out[key] = fo.get_tensor(name)
            elif kind == "idx":
                codes = base_sd[key].clone()
                flat = codes.view(-1)
                at = fo.get_tensor(name).long()
                flat[at] = (flat[at].int() + fo.get_tensor(f"delta.{key}").int()).to(torch.int8)
                out[key] = codes
    if gate_scale != 1.0:
        scaled = [k for k in out if k.endswith(GATE_SCALE_SUFFIX)]
        if not scaled:
            raise ValueError(f"gate_scale {gate_scale} has no to_gate_compress tensors to scale")
        for key in scaled:
            out[key] = (out[key].float() * gate_scale).to(out[key].dtype)
    return out
