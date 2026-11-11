"""The MiniMax-H3-Image release as tensors, and the three places a ComfyUI file differs from it.

The release's `transformer/` folder already uses the tensor names core's model
uses (`blocks.N.attn.qkv_proj.weight`, ...), unlike the H3 video release's
diffusers-named `transformer/`. A single file ComfyUI loads differs from the
release shards in exactly three places, each read from core rather than assumed:

1. **Fused qkv row order.** The release stores `qkv_proj.weight` per head,
   `[head][q | k | v]`; core splits the projection as `[q_all | k_all | v_all]`
   (`comfy/ldm/minimax/model.py`, `Attention.forward`) and nothing on its load
   path permutes. `comfy_qkv_order` is that permutation, for the block and the
   token-refiner projections alike.
2. **The output heads.** The release keeps one plain output layer in
   `transformer/` and six per-step heads in `pdd_fused_heads.safetensors`. Core
   reads one stacked `final_layer.video_out` whose first row block is a whole
   head and whose later blocks are offsets from it
   (`comfy/ldm/minimax/model.py::_pdd_head`). `stack_heads` builds that.
3. **The sampling grid** rides in the file's metadata
   (`config.transformer.pdd_steps` and `pdd_sigmas`), which
   `comfy/model_detection.py` reads; the release keeps it in `pdd_config.json`.

Everything else is copied byte for byte, dtype included. The timestep embedder
and the full-width AdaLN projections are kept, so nothing is approximated: a
pruned file replaces those with a small table, which this one does not.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open

HEADS = 56
HEAD_DIM = 128

_ST_DTYPE = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16",
             torch.int8: "I8", torch.uint8: "U8", torch.int64: "I64"}


class Sharded:
    """Tensors of a sharded safetensors folder, read by its index one at a time."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        index = next(self.folder.glob("*.safetensors.index.json"))
        self.where: dict[str, str] = json.loads(index.read_text())["weight_map"]
        self._open: dict[str, object] = {}

    def keys(self) -> list[str]:
        return sorted(self.where)

    def get(self, key: str) -> torch.Tensor:
        shard = self.where[key]
        if shard not in self._open:
            self._open[shard] = safe_open(str(self.folder / shard), "pt")
        return self._open[shard].get_tensor(key)

    def close(self) -> None:
        self._open.clear()


def comfy_qkv_order(weight: torch.Tensor, heads: int = HEADS, head_dim: int = HEAD_DIM) -> torch.Tensor:
    """Release rows `[head][q|k|v]` -> core rows `[q_all|k_all|v_all]`."""
    if weight.shape[0] != heads * 3 * head_dim:
        raise ValueError(f"a fused qkv weight has {heads * 3 * head_dim} rows, got {weight.shape[0]}")
    return weight.reshape(heads, 3, head_dim, -1).permute(1, 0, 2, 3).reshape(weight.shape).contiguous()


def stack_heads(heads: torch.Tensor) -> torch.Tensor:
    """`[steps, out, ...]` whole heads -> core's bank: head 0 whole, then each later head minus head 0."""
    heads = heads.to(torch.float32)
    bank = torch.cat([heads[:1], heads[1:] - heads[:1]], dim=0)
    return bank.reshape(heads.shape[0] * heads.shape[1], *heads.shape[2:]).contiguous()


def is_fused_qkv(key: str) -> bool:
    return key.endswith("attn.qkv_proj.weight")


def comfy_tensor(key: str, release: Sharded, heads_file) -> torch.Tensor:
    """One tensor of the single file, from the release."""
    if key == "final_layer.video_out.weight":
        return stack_heads(heads_file.get_tensor("video_out.weight"))
    if key == "final_layer.video_out.bias":
        return stack_heads(heads_file.get_tensor("video_out.bias"))
    tensor = release.get(key)
    return comfy_qkv_order(tensor) if is_fused_qkv(key) else tensor


def metadata(pdd_config: dict) -> dict[str, str]:
    """The file metadata core reads the head count and the sampling grid from."""
    sigmas = [float(s) for s in pdd_config["sigmas"]]
    if len(sigmas) != int(pdd_config["num_inference_steps"]) + 1:
        raise ValueError("pdd_config.json: the grid needs one more point than it has steps")
    return {"config": json.dumps({"transformer": {"pdd_steps": len(sigmas) - 1, "pdd_sigmas": sigmas}})}


def raw_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.contiguous()
    if tensor.dtype == torch.bfloat16:   # numpy has no bf16: the same bytes as int16
        tensor = tensor.view(torch.int16)
    return tensor.numpy().tobytes()


def write_safetensors(path: str | Path, entries: list[tuple[str, str, tuple[int, ...]]], get, meta: dict[str, str],
                      progress=None) -> int:
    """Write a safetensors file one tensor at a time.

    `entries` is `(key, dtype, shape)` in file order, known before any tensor is
    read, so the header is written first and the file never sits in memory.
    `get(key)` returns the tensor; a tensor whose dtype or shape disagrees with
    its entry is an error, never a silent cast.
    """
    width = {"F32": 4, "BF16": 2, "F16": 2, "I8": 1, "U8": 1, "I64": 8}
    header: dict[str, object] = {"__metadata__": meta}
    offset = 0
    for key, dtype, shape in entries:
        size = width[dtype]
        for dim in shape:
            size *= dim
        header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for i, (key, dtype, shape) in enumerate(entries):
            tensor = get(key)
            if _ST_DTYPE[tensor.dtype] != dtype or tuple(tensor.shape) != tuple(shape):
                raise ValueError(f"{key}: planned {dtype} {tuple(shape)}, got {_ST_DTYPE[tensor.dtype]} {tuple(tensor.shape)}")
            out.write(raw_bytes(tensor))
            if progress is not None:
                progress(i, key)
    return 8 + len(blob) + offset


def plan(release: Sharded, heads_file) -> list[tuple[str, str, tuple[int, ...]]]:
    """`(key, dtype, shape)` of every tensor of the single file, without reading a weight."""
    entries = []
    for key in release.keys():
        shard = release.where[key]
        if shard not in release._open:
            release._open[shard] = safe_open(str(release.folder / shard), "pt")
        view = release._open[shard].get_slice(key)
        dtype, shape = view.get_dtype(), tuple(view.get_shape())
        if key.startswith("final_layer.video_out."):
            head = heads_file.get_slice("video_out." + key.rsplit(".", 1)[1])
            hshape = tuple(head.get_shape())
            dtype, shape = "F32", (hshape[0] * hshape[1], *hshape[2:])
        entries.append((key, dtype, shape))
    release.close()
    return entries
