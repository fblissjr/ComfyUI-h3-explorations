"""Capture real q/k/v from an H3 render, for kernel-divergence work.

Exists because accuracy numbers measured on `torch.randn` are a pessimistic
bound rather than an estimate. The sage fork measured the fp8-vs-fp16 gap at
2.7x on synthetic input and 1.3x on real captured activations -- a 2x error in
the number this repo cites to justify running fp16. Real captures are the only
way to settle that, and the previous capture set was not kept.

**Inert unless `H3_CAPTURE` is set.** No node input, no schema change, no graph
change: this must not be reachable by opening a workflow, because a capture at
H3's real length writes gigabytes per call.

    H3_CAPTURE="dir=/path,blocks=0:24:49,steps=3:11" ~/ComfyUI/start.sh

  dir     where to write. Required.
  blocks  colon-separated block indices, in first-seen call order. Default 0.
  steps   colon-separated step indices, 0-based. Default 1 -- NOT 0, whose
          activation statistics are not representative of the trajectory.

Writes `qkv_L{length}_S{seq}_b{block}_s{step}.pt`, each a
`{"q","k","v"}` dict of bf16 `[B, H, S, D]`. Block and step are recoverable
from the filename, because the number this replaces has no recorded
provenance beyond "one block, one step" and that is the whole problem.

## Two things about the tensors, both deliberate

**Layout is transposed on the way out.** The forward holds q/k/v as
`[1, S, H, D]` (three views of one fused qkv buffer); consumers of these
captures want `[B, H, S, D]`. The transpose is a view of identical data, made
contiguous at save time, so nothing is altered -- but the file is HND and the
call site is NHD, and confusing the two silently transposes an accuracy
measurement.

**Captured AFTER the fused RMSNorm+RoPE**, which is what sage actually
receives. Capturing before it would measure a tensor no kernel ever sees.

## Why sage, and why Sol must be off

These are the tensors the *sage* path holds. With Sol-Attn on, sage receives
nothing -- H3's DiT has one attention site and Sol takes it -- so a capture
run must have Sol bypassed or it will produce no files and look like a broken
hook rather than an empty one. The block counter warns when that happens.
"""

from __future__ import annotations

import os
import re
import threading

_SPEC = os.environ.get("H3_CAPTURE", "")
enabled = bool(_SPEC)

_lock = threading.Lock()
_block_of: dict[int, int] = {}      # id(module) -> index, in first-seen order
_calls: dict[int, int] = {}         # block index -> times seen (i.e. step)
_written: set[tuple[int, int]] = set()
_config: dict = {}


def _parse(spec):
    out = {"dir": None, "blocks": {0}, "steps": {1}}
    for part in spec.split(","):
        key, _, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if key == "dir":
            out["dir"] = val
        elif key in ("blocks", "steps") and val:
            out[key] = {int(x) for x in re.split(r"[:;]", val) if x.strip()}
    return out


if enabled:
    _config = _parse(_SPEC)
    if not _config["dir"]:
        enabled = False
        print("[h3_capture] H3_CAPTURE set but no dir=; capture disabled")
    else:
        os.makedirs(_config["dir"], exist_ok=True)
        print(f"[h3_capture] ARMED: dir={_config['dir']} "
              f"blocks={sorted(_config['blocks'])} steps={sorted(_config['steps'])}")


def maybe_capture(module, q, k, v, length_hint=None):
    """Save this call's q/k/v if it matches the requested (block, step).

    Called from the sage forward after rope. Cheap and returns immediately
    when disabled, which is every normal render.

    Block index is assigned by first-seen order rather than read off the
    module, because the attention modules carry no index and a patch-time tag
    would put a capture concern into the patching loop. Blocks run 0..N-1 in
    order every step, so first-seen order IS the block order; the step is then
    just how many times that block has been seen.
    """
    if not enabled:
        return
    import torch

    key = id(module)
    with _lock:
        if key not in _block_of:
            _block_of[key] = len(_block_of)
        block = _block_of[key]
        step = _calls.get(block, 0)
        _calls[block] = step + 1
        if block not in _config["blocks"] or step not in _config["steps"]:
            return
        if (block, step) in _written:
            return
        _written.add((block, step))

    # [1, S, H, D] -> [B, H, S, D]. contiguous() because the consumer slices
    # heads and expects them to be the outer stride.
    qh, kh, vh = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    seq = qh.shape[2]
    name = (f"qkv_L{length_hint if length_hint is not None else 'na'}"
            f"_S{seq}_b{block}_s{step}.pt")
    path = os.path.join(_config["dir"], name)
    torch.save({"q": qh, "k": kh, "v": vh}, path)
    size = os.path.getsize(path) / 2**30
    print(f"[h3_capture] wrote {name}  {tuple(qh.shape)} {qh.dtype}  {size:.2f} GiB")


def summary():
    """What was seen, for confirming a capture run did what was asked."""
    if not enabled:
        return "capture disabled"
    if not _block_of:
        return ("[h3_capture] NO CALLS SEEN. The sage forward never ran -- with "
                "Sol-Attn on it takes the only DiT attention call and sage gets "
                "nothing. Bypass Sol and re-run.")
    return (f"[h3_capture] {len(_block_of)} blocks seen, "
            f"{max(_calls.values())} steps, {len(_written)} files written")
