"""Capture real q/k/v from an H3 render, for kernel-divergence work.

Exists because accuracy numbers measured on `torch.randn` are a pessimistic
bound rather than an estimate. Every fp8-vs-fp16 accuracy ratio this repo used
to carry was **withdrawn on 2026-08-16 as untrusted** (`docs/evidence.md`) --
the synthetic sweep measures an input distribution H3 does not have, and the
competing real-activation figure was never re-derived here. Real captures are
the only way to settle it, and no kernel has yet been graded against the ones
this script produced.

**Inert unless `H3_CAPTURE` is set.** No node input, no schema change, no graph
change: this must not be reachable by opening a workflow, because a capture at
H3's real length writes gigabytes per call.

    H3_CAPTURE="dir=/path,blocks=0:24:49,steps=3:11" <comfy>/start.sh

  dir     where to write. Required.
  blocks  colon-separated block indices, in first-seen call order. Default 0.
  steps   colon-separated step indices, 0-based. Default 1 -- NOT 0, whose
          activation statistics are not representative of the trajectory.
  cycle   sampler steps per render, so a second render in the same server
          process is recognised as a new one. **Declare it or omit it; it is
          never inferred**, because the step count varies per graph (16 base,
          8 and 4 turbo) and Sol-Attn narrows what sage sees to its sigma
          window. Omitted means no boundary is detected: the counter keeps
          rising and a second render captures nothing, which `summary()` says.

  pre     `1` also writes the FUSED qkv projection from before RMSNorm and
          RoPE (`qkvpre_*.pt`, beside the usual file); `only` writes it
          INSTEAD of the post-RoPE file. For
          `bench/grade_sol_impl_on_capture.py`, which grades ComfyUI core's
          chunked-producer Sol path against ours; see `maybe_capture_pre`.
          Off by default, and off leaves every other file byte-identical to
          what this module wrote before the key existed (2026-09-10).

Filenames from the second render onward carry `_r{n}`. The first render's names
are unchanged, so existing captures and every glob over them still match.

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

These are the tensors the *sage* path holds, and **a capture run must have
Sol-Attn bypassed** -- but not for the reason this docstring gave until
2026-08-14. It said sage receives nothing with Sol on, so the run would
produce no files. That is the claim retracted in `docs/SOLATTN.md`: Sol only
takes the calls inside its sigma window, so at the shipped `0.2 / 0.9` and 16
steps sage still runs **5 of 16** steps.

The real failure is worse than an empty directory, because it is not empty.
Those 5 steps are 0-3 and 15 -- both ends of the schedule and none of the
middle -- so a capture taken with Sol on yields a plausible-looking set of
files drawn from an unrepresentative slice of the trajectory, and every
accuracy number computed from it inherits that skew silently. An empty run
announces itself; this one does not. The block counter still warns.
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
_written: set[tuple[int, int, int]] = set()   # (render, block, step) already saved
_render = 0                         # render index within this server process
_final_step = 0                     # forwards seen by the final-layer tap
_config: dict = {}


def _parse(spec):
    out = {"dir": None, "blocks": {0}, "steps": {1}, "cycle": None, "final": False,
           "pre": None}
    for part in spec.split(","):
        key, _, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if key == "dir":
            out["dir"] = os.path.expanduser(val)
        elif key == "cycle" and val:
            out["cycle"] = int(val)
        elif key == "final" and val:
            out["final"] = val.strip().lower() not in ("0", "false", "no", "off")
        elif key == "pre" and val:
            # "only" suppresses the post-RoPE file; any other truthy value
            # writes the pre-norm file BESIDE it. See maybe_capture_pre.
            v = val.strip().lower()
            out["pre"] = ("only" if v == "only"
                          else None if v in ("0", "false", "no", "off") else "both")
        elif key in ("blocks", "steps") and val:
            out[key] = {int(x) for x in re.split(r"[:;]", val) if x.strip()}
    return out


def _sync_spec():
    global _SPEC, enabled, _config
    spec = os.environ.get("H3_CAPTURE", "")
    if not _config or spec != _SPEC:
        _SPEC = spec
        enabled = bool(_SPEC)
        if enabled:
            _config = _parse(_SPEC)
            if not _config["dir"]:
                enabled = False
                print("[h3_capture] H3_CAPTURE set but no dir=; capture disabled", flush=True)
            else:
                os.makedirs(_config["dir"], exist_ok=True)
                # flush=True on every print here: core's LogInterceptor
                # (app/logger.py) rewraps stdout block-buffered when it is a
                # pipe, so without it this line reached a piped server log
                # only when a later logging call flushed, mid-render, and a
                # launcher waiting for it before submitting waited forever.
                print(f"[h3_capture] ARMED: dir={_config['dir']} "
                      f"blocks={sorted(_config['blocks'])} steps={sorted(_config['steps'])}"
                      + (" final=on" if _config.get("final") else "")
                      + (f" pre={_config['pre']}" if _config.get("pre") else ""),
                      flush=True)
        else:
            _config = {}


_sync_spec()


def maybe_capture(module, q, k, v, length_hint=None, kernel="sage",
                  transformer_options=None):
    """Save this call's q/k/v if it matches the requested (block, step).

    Called from the sage forward after rope. Cheap and returns immediately
    when disabled, which is every normal render.

    **`kernel` says WHICH kernel then took this call, and it is not
    decoration.** Every shipped video graph wires sage AND Sol, and the two
    split the calls between them: Sol takes the DiT blocks inside its sigma
    window that are not in `dense_blocks`, sage takes the rest -- the two
    token-refiner calls below `min_tokens`, the dense blocks, and every step
    outside the band. A capture with no tag cannot say which of those it
    holds, and until 2026-08-30 it could not hold the Sol ones at ALL: this
    function is called from our sage forward, and Sol's composition routes
    around that forward on the calls it takes, so a capture on a shipped graph
    silently recorded exactly the complement of Sol's work. See
    `attention.py`'s `sol_take_forward`, which is what closes it.

    **`transformer_options` is read for the packed layout's segment
    boundaries only**, and nothing else. `[text | cond | ref | audio | video]`
    have genuinely different activation statistics, and without the boundaries
    a consumer can only bin by position and hope -- which is why
    `bench/grade_sage_on_capture.py` samples positional strata and cannot
    answer whether error concentrates at a segment edge. Recording them costs
    a tuple. Requested independently by three lanes on 2026-08-30.

    Block index is assigned by first-seen order rather than read off the
    module, because the attention modules carry no index and a patch-time tag
    would put a capture concern into the patching loop. Blocks run 0..N-1 in
    order every step, so first-seen order IS the block order; the step is then
    just how many times that block has been seen.
    """
    global _render
    _sync_spec()
    if not enabled:
        return
    import torch

    with _lock:
        # Block index, render boundary and step counter live in `_block_step`
        # since 2026-09-10, shared with `maybe_capture_pre`, which reads the
        # same indices without advancing them. The behaviour here is unchanged.
        block, step = _block_step(module, advance=True)
        if block not in _config["blocks"] or step not in _config["steps"]:
            return
        # `pre=only`: this call is still COUNTED above, so the step index
        # stays aligned with `maybe_capture_pre`, but no post-RoPE file.
        if _config.get("pre") == "only":
            return
        # Keyed by render, and `_written` is never cleared. Clearing it let a
        # second render silently overwrite the first render's files -- multiple
        # GiB destroyed with no prompt, and any manifest checksum already
        # computed for them going stale with nothing noticing.
        if (_render, block, step) in _written:
            return
        _written.add((_render, block, step))
        render = _render

    # [1, S, H, D] -> [B, H, S, D]. contiguous() because the consumer slices
    # heads and expects them to be the outer stride.
    #
    # **Copy to CPU BEFORE transpose/contiguous, not after.** Doing it on the
    # device allocates three [1,H,S,D] buffers next to a model that is already
    # near the card's limit: 5.4 GiB at S=124,582 (2 image refs, 362f), against
    # ~6.7 GiB headroom at that size. The old order OOMed the render rather than
    # the capture, so it would have looked like a length limit, not a tooling
    # one. `q` arrives contiguous, so `.cpu()` is a straight copy and the
    # transpose then costs host memory, which is not the scarce resource.
    # Saved bytes are identical either way; only where the intermediate lives
    # changes. Changed 2026-08-16 -- captures before that date used the old
    # order, which is why they are all 124f.
    qh, kh, vh = (t.cpu().transpose(1, 2).contiguous() for t in (q, k, v))
    seq = qh.shape[2]
    # `_r{n}` appears only from the SECOND render onward, so first-render
    # filenames are unchanged and every existing glob and capture directory
    # keeps matching. Without it a second render collides with the first on
    # every name.
    suffix = f"_r{render}" if render else ""
    # The kernel goes in the FILENAME as well as the payload. A consumer
    # globbing a directory should not have to load a multi-GiB tensor to find
    # out which arm a file belongs to -- that is the same mistake as reading
    # arm identity off a render's filename, which cost a wrong pair earlier
    # today. `_ksage` / `_ksol`, absent when the tag is the historical default,
    # so pre-2026-08-30 captures keep matching every existing glob.
    ktag = "" if kernel == "sage" else f"_k{kernel}"
    name = (f"qkv_L{length_hint if length_hint is not None else 'na'}"
            f"_S{seq}_b{block}_s{step}{ktag}{suffix}.pt")
    path = os.path.join(_config["dir"], name)
    # Segment bounds, when the layout published them. `sol_h3_video_span` is
    # what the Sol node's rope hook publishes; the full table is preferred and
    # the span is the fallback, so a capture taken with Sol absent still says
    # where video starts rather than saying nothing.
    segments = None
    if isinstance(transformer_options, dict):
        segments = transformer_options.get("h3_segments")
        if segments is None:
            span = transformer_options.get("sol_h3_video_span")
            audio = transformer_options.get("sol_h3_audio_span")
            if span is not None:
                segments = [(int(span[0]), int(span[1]), "video")]
                if audio is not None:
                    segments.insert(0, (int(audio[0]), int(audio[1]), "audio"))

    # **Index fields are TOP-LEVEL SCALARS, not filename-encoded**, so a
    # consumer joins on a dict lookup instead of writing its own parser. Asked
    # for by the PDD lane 2026-08-30, whose own observations key on
    # `block` / `module` / `sigma`: with these fields the two capture formats
    # taken from ONE render join exactly -- same seed, same trajectory -- and
    # without them every consumer re-derives the index from a name and they
    # drift. The filename keeps carrying them too, for globbing; it is the
    # convenience copy, and this is the authority.
    sigma = None
    if isinstance(transformer_options, dict):
        sigmas = transformer_options.get("sigmas")
        if sigmas is not None:
            try:
                sigma = float(sigmas[0])
            except (TypeError, IndexError, ValueError):
                sigma = None

    # **The weights this capture came from, when they are not the shipped
    # ones.** `MiniMaxH3PDDLoRA`'s un-merged path applies the LoRA delta at the
    # call instead of folding it into the quantised weight, which leaves the
    # stored weight measurably closer to bf16 than the shipped configuration --
    # measured in that lane at 0.00942 against 0.01058. A capture taken on a
    # joint render with that path on is therefore graded on activations from a
    # slightly different model, and that is a caveat about GENERALISATION
    # rather than a confound: kernels are compared against each other on the
    # SAME activations, so the model is upstream of the comparison. Recorded
    # here so the file carries the caveat instead of someone having to
    # remember which render it came from. Key published by pdd_lora.py.
    unmerged = None
    if isinstance(transformer_options, dict):
        unmerged = transformer_options.get("minimax_h3_unmerged_blocks")

    record = {"q": qh, "k": kh, "v": vh,
              "kernel": kernel, "block": int(block), "step": int(step),
              "sigma": sigma, "seq_len": int(seq), "render": int(render),
              "unmerged_blocks": unmerged,
              # The render this belongs to. ComfyUI's executing context (see
              # provenance.py) names it; it is the join to /history's output
              # filenames and to the Sol route record's render row.
              "prompt_id": _prompt_id(),
              # The process that wrote this: its launch flags (`--fast
              # fp16_accumulation` changes the numerics a capture holds) and
              # the versions under it. The manifest generator copies this
              # into `provenance.server`; nothing offline can recover it.
              "server": _server_stamp()}
    if segments is not None:
        record["segments"] = segments
    # Token order. With Sol's reorder on, q/k/v are captured in PERMUTED row
    # order while `segments` above describes the raster layout, and until
    # 2026-09-17 nothing in the file said which. A consumer that permutes a
    # capture itself (bench/sweep_sol_orderings_on_capture.py) needs a raster
    # one, and can now check instead of trusting the caller.
    if isinstance(transformer_options, dict):
        record["sol_morton"] = bool(transformer_options.get("sol_morton", False))
        if record["sol_morton"]:
            record["sol_morton_curve"] = str(transformer_options.get("sol_morton_curve", "3d"))

    # **The capture asserts its own shape before it is written.** Adopted from
    # the PDD lane, which hit two silent short-capture bugs in one day: a file
    # of the wrong length is not detectable later, because nothing downstream
    # knows what length it should have been. Cheap, and it fails at write time
    # where the cause is still on screen.
    # `field`, not `name`: this loop used to rebind the filename, so every
    # write logged as "wrote v" (found 2026-09-03 on the first Base16 capture;
    # the file itself was always named correctly, only the log lied).
    for field, t in (("q", qh), ("k", kh), ("v", vh)):
        if t.shape != qh.shape or t.ndim != 4 or t.shape[2] != seq:
            raise RuntimeError(
                f"h3_capture: {field} is {tuple(t.shape)}, expected "
                f"{tuple(qh.shape)} with {seq} rows. Refusing to write a "
                f"capture whose shape nothing downstream could check.")
    if segments is not None and segments[-1][1] != seq:
        raise RuntimeError(
            f"h3_capture: segments end at {segments[-1][1]} but the sequence "
            f"is {seq} rows. A boundary table that does not cover the capture "
            f"would silently mis-bin every consumer that trusts it.")

    torch.save(record, path)
    size = os.path.getsize(path) / 2**30
    print(f"[h3_capture] wrote {name}  {tuple(qh.shape)} {qh.dtype}  "
          f"kernel={kernel}  segments={'yes' if segments else 'NO'}  "
          f"{size:.2f} GiB", flush=True)


def _block_step(module, advance):
    """(block, step) of this call. Call with `_lock` held.

    Shared since 2026-09-10 by `maybe_capture`, which ADVANCES the per-block
    counter once per call, and `maybe_capture_pre`, which runs earlier in the
    same forward and only reads it, so the two files of one call carry the
    same indices and join exactly. The render boundary is idempotent within a
    call: once it fires the counter is cleared, so the second caller of the
    same forward cannot fire it again.
    """
    global _render
    # Prefer the index the patching loop stamped on the module. First-seen
    # ordering below is the fallback and it is WRONG ACROSS A MODEL SWAP: the
    # ids belong to the modules of whichever checkpoint was loaded, so the
    # first render after a swap assigns 50..99 to the new blocks, no requested
    # index matches, `block == 0` never fires again, and the render counter
    # jams. Capture then stops silently for the rest of the process.
    #
    # Found 2026-08-21 on the open-experiment-22 arms, which swap checkpoints
    # between renders by design: the first two arms captured, the remaining
    # nine wrote nothing and the only symptom was empty directories. The
    # module comment used to argue against a patch-time tag because it "would
    # put a capture concern into the patching loop". It is worth that -- the
    # alternative was an instrument that quietly stops.
    tagged = getattr(module, "_h3_block_index", None)
    key = id(module)
    if tagged is not None:
        _block_of[key] = tagged
    elif key not in _block_of:
        _block_of[key] = len(_block_of)
    block = _block_of[key]

    # Render boundary. `cycle` is DECLARED, never guessed, and the default
    # is no reset at all.
    #
    # It was hardcoded to 16 until 2026-08-17 -- a second copy of
    # `h3_config.SAMPLING["steps"]`, and wrong for everything else this repo
    # ships. At 20 steps it fired MID-render, so real steps 16-19 were
    # recorded as 0-3 and a file named `_s3` ended up holding step 19: a
    # corrupted capture whose filename lied. Below 16 it never fired
    # (`TURBO_STEPS` is 8, `TURBO_768P_STEPS` is 4), so a second render in
    # the same server process kept counting upward and captured nothing.
    #
    # Nothing here can infer it: the step count varies per graph, and with
    # Sol-Attn on, sage sees only the steps inside its sigma window rather
    # than all of them. A guess is guaranteed wrong for somebody, so
    # `cycle=` in `H3_CAPTURE` is how it gets stated.
    if _config.get("cycle") and block == 0 and _calls.get(0, 0) >= _config["cycle"]:
        _render += 1
        _calls.clear()
        # The final tap counts forwards on its own axis, so the render
        # boundary has to reset it here too or a second render's velocity
        # lands at a step index no filter matches and nothing is written.
        globals()["_final_step"] = 0

    step = _calls.get(block, 0)
    if advance:
        _calls[block] = step + 1
    return block, step


def _chunk_check(module, x, qkv):
    """Whether core's first producer chunk, projected alone, equals those rows
    of the full projection. `maybe_capture_pre` says why it is measured.

    Compared in row slices so the check itself allocates next to nothing on a
    card the model already fills (the reason `maybe_capture` copies to host
    before reshaping)."""
    try:
        from comfy_extras.nodes_sparse_attention import PRODUCER_CHUNK
    except Exception as exc:                           # noqa: BLE001 -- absent is a value
        return {"skipped": f"core's PRODUCER_CHUNK unavailable: {type(exc).__name__}"}
    import torch
    n = min(int(PRODUCER_CHUNK), int(x.shape[0]))
    with torch.no_grad():
        part = module.qkv_proj(x[:n])
        identical, worst = True, 0.0
        for i in range(0, n, 512):
            a, b = part[i:i + 512], qkv[i:min(i + 512, n)]
            identical = identical and bool(torch.equal(a, b))
            worst = max(worst, float((a.float() - b.float()).abs().max()))
    del part
    return {"rows": n, "producer_chunk": int(PRODUCER_CHUNK),
            "identical": identical, "max_abs_diff": worst}


def maybe_capture_pre(module, qkv, x, rope_freqs, transformer_options=None,
                      length_hint=None):
    """Save this call's FUSED qkv projection, before RMSNorm and RoPE.

    Armed by `pre=` in `H3_CAPTURE` and inert otherwise: `pre=1` writes this
    beside the usual post-RoPE file, `pre=only` instead of it. Exists for
    `bench/grade_sol_impl_on_capture.py`. ComfyUI core's own Sol node
    (`comfy_extras/nodes_sparse_attention.py`) does not take the post-RoPE
    tensors the rest of this module records: its H3 path feeds kitchen's
    `sol_attn_chunked` the `qkv_proj` output in row chunks and applies the q/k
    RMSNorm and RoPE inside the kernel. Grading it needs the tensor from BEFORE
    the in-place norm this forward runs next, plus the rope table, the norm
    weights and the eps it used.

    **Must be called before `rms_rope_split_half_`**: that op rewrites the q
    and k columns of this same buffer in place, which is why the copy to host
    happens here and not later.

    Reads the (block, step) of this call without advancing the counter;
    `maybe_capture`, later in the same forward, advances it. So a `qkvpre_`
    file and a `qkv_` file of one call carry the same indices.

    Also records, for the grader:
      segments        the packed layout's (start, stop, kind) table, from the
                      `minimax_h3_layout` core publishes on every forward (since
                      Comfy-Org/ComfyUI#16072), so it is present with Sol
                      absent, where `maybe_capture` records none;
      uuids, cond_or_uncond  which conditioning branch this call was, since
                      core carries its statistics per branch;
      chunk_check     whether projecting core's first `PRODUCER_CHUNK` rows
                      alone reproduces those rows of the full projection. The
                      graded core arm is fed chunks of THIS full projection,
                      which are core's own bytes only if the linear is
                      row-independent (kitchen's int8 path quantises
                      activations per row, `comfy_kitchen/tensor/int8.py`).
                      Measured on the real weights rather than assumed; the
                      first chunk only, to keep the cost to one extra slice.

    The norm weights are captured, not read from the checkpoint at grade time:
    they are a few hundred bytes, and capturing them records what this process
    actually passed (after `cast_to`) instead of what a file says.
    """
    _sync_spec()
    if not enabled or not _config.get("pre") or rope_freqs is None:
        return
    import torch
    import comfy.model_management

    with _lock:
        block, step = _block_step(module, advance=False)
        if block not in _config["blocks"] or step not in _config["steps"]:
            return
        if (_render, block, step, "pre") in _written:
            return
        _written.add((_render, block, step, "pre"))
        render = _render

    heads, head_dim = int(module.heads), int(module.head_dim)
    seq = int(qkv.shape[0])
    if qkv.ndim != 2 or qkv.shape[1] != 3 * heads * head_dim:
        raise RuntimeError(
            f"h3_capture: fused qkv is {tuple(qkv.shape)}, expected [S, {3 * heads * head_dim}]. "
            f"Refusing to write a pre-norm capture nothing downstream could check.")
    # Host copies FIRST: the in-place norm that follows this call rewrites q and k.
    qkv_h = qkv.detach().cpu()
    qw = comfy.model_management.cast_to(module.q_norm.weight, device=qkv.device).detach().cpu()
    kw = comfy.model_management.cast_to(module.k_norm.weight, device=qkv.device).detach().cpu()
    rope_h = rope_freqs.detach().cpu()
    chunk_check = _chunk_check(module, x, qkv)

    to = transformer_options if isinstance(transformer_options, dict) else {}
    segments, signature = None, None
    layout = to.get("minimax_h3_layout")
    if layout is not None and getattr(layout, "seq_len", None) == seq:
        segments = [(int(a), int(b), str(k)) for a, b, k in layout.segments]
        signature = [int(v) for v in (getattr(layout, "signature", ()) or ())]
    sigma = None
    sigmas = to.get("sigmas")
    if sigmas is not None:
        try:
            sigma = float(sigmas[0])
        except (TypeError, IndexError, ValueError):
            sigma = None
    if segments is not None and segments[-1][1] != seq:
        raise RuntimeError(
            f"h3_capture: layout segments end at {segments[-1][1]} but the fused qkv has "
            f"{seq} rows; refusing to write a boundary table that mis-bins every consumer.")

    record = {"kind": "qkv_pre", "qkv": qkv_h, "rope_freqs": rope_h,
              "q_norm_weight": qw, "k_norm_weight": kw,
              "rope_eps": float(module.q_norm.eps), "rot_dim": int(rope_freqs.shape[-3] * 2),
              "heads": heads, "head_dim": head_dim,
              "block": int(block), "step": int(step), "sigma": sigma, "seq_len": seq,
              "render": int(render), "segments": segments, "layout_signature": signature,
              "uuids": [str(u) for u in (to.get("uuids") or [])] or None,
              "cond_or_uncond": [int(c) for c in (to.get("cond_or_uncond") or [])] or None,
              "chunk_check": chunk_check,
              "unmerged_blocks": to.get("minimax_h3_unmerged_blocks"),
              "prompt_id": _prompt_id(), "server": _server_stamp()}
    suffix = f"_r{render}" if render else ""
    name = (f"qkvpre_L{length_hint if length_hint is not None else 'na'}"
            f"_S{seq}_b{block}_s{step}{suffix}.pt")
    path = os.path.join(_config["dir"], name)
    torch.save(record, path)
    size = os.path.getsize(path) / 2**30
    print(f"[h3_capture] wrote {name}  qkv{tuple(qkv_h.shape)} {qkv_h.dtype}  "
          f"segments={'yes' if segments else 'NO'}  chunk identical="
          f"{chunk_check.get('identical', 'n/a')}  {size:.2f} GiB", flush=True)


def wants_final():
    """Whether `final=1` was declared. Read by the patching node."""
    _sync_spec()
    return bool(enabled and _config.get("final"))


def _prompt_id():
    try:
        from comfy_execution.utils import get_executing_context
        ctx = get_executing_context()
        return getattr(ctx, "prompt_id", None) if ctx else None
    except Exception:                                  # noqa: BLE001 -- unknown is a value; the join is best effort
        return None


def _server_stamp() -> dict:
    import sys
    # Imported HERE: this module imports torch only inside functions, and
    # until 2026-09-10 this one used the name without importing it, so the
    # first record written after the stamp arrived (dae6864, 2026-09-03)
    # would have raised NameError mid-render. No capture had run since.
    import torch
    # str(): `torch.__version__` is a TorchVersion, which a weights_only load
    # refuses as an unknown global; a plain string keeps every record loadable
    # with `torch.load(..., weights_only=True)`.
    out = {"argv": list(sys.argv), "torch": str(torch.__version__),
           "cuda": torch.version.cuda, "comfy_kitchen": None, "pid": os.getpid()}
    try:
        import importlib.metadata as _md
        out["comfy_kitchen"] = _md.version("comfy_kitchen")
    except Exception:                                  # noqa: BLE001 -- absent is a value here
        pass
    # EFFECTIVE settings, not only argv: the parsed ComfyUI namespace, every
    # JSON-safe scalar of it, so `--fast fp16_accumulation` and the dtype
    # forcings are read as the process resolved them.
    try:
        from comfy.cli_args import args as _args
        out["comfy_args"] = {k: v for k, v in sorted(vars(_args).items())
                             if isinstance(v, (bool, int, float, str, type(None)))}
        fast = getattr(_args, "fast", None)
        out["comfy_args"]["fast"] = sorted(str(x) for x in fast) if fast else fast
    except Exception as exc:                           # noqa: BLE001
        out["comfy_args"] = {"error": str(exc)}
    # Sage decides the dense trajectory a capture holds, so its build identity
    # goes in: version, the file it imports from, and that tree's git head
    # when it is a checkout (this box runs an editable fork).
    try:
        import subprocess
        import sageattention as _sa
        import importlib.metadata as _md
        path = os.path.dirname(os.path.abspath(_sa.__file__))
        head = subprocess.run(["git", "-C", path, "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, timeout=5)
        out["sageattention"] = {"version": _md.version("sageattention"), "path": path,
                                "git_head": head.stdout.strip() if head.returncode == 0 else None}
    except Exception as exc:                           # noqa: BLE001
        out["sageattention"] = {"error": str(exc)}
    return out


def maybe_capture_final(out, length_hint=None):
    """Save the DiT's own output -- the velocity -- at the requested steps.

    This is the one tensor that is "the network output", which is what
    `docs/open_experiments.md` #22 compares between a pruned and an unpruned
    checkpoint. q/k/v at five depths say where two forwards diverge; only this
    says by how much it mattered by the end.

    **Counts forwards on its own axis rather than reading the block counter.**
    Deriving the step from `_calls[0]` would work only while the sage forward
    is the one running every block, which is exactly the assumption that fails
    with Sol-Attn on -- Sol takes the calls inside its sigma window and sage
    sees the rest, so the block counter undercounts and the velocity would be
    filed under the wrong step. An independent counter cannot drift for that
    reason. It shares `steps=`, `_written` and the render index with the q/k/v
    path so the two stay aligned on everything else.

    Filed under block -1 in `_written`, which no real block can collide with.
    """
    global _final_step
    _sync_spec()
    if not enabled or not _config.get("final"):
        return
    import torch

    with _lock:
        step = _final_step
        _final_step += 1
        if step not in _config["steps"]:
            return
        if (_render, -1, step) in _written:
            return
        _written.add((_render, -1, step))
        render = _render

    # H3 is an AUDIO-VIDEO model and its forward returns BOTH velocities as a
    # list -- `[-video_out, -audio_out]` (comfy/ldm/minimax/model.py:732). The
    # first version of this tap expected a bare tensor, refused the list and
    # wrote nothing, which is the refusal working but the wrong expectation.
    # Both streams are kept: the video one is what #22 compares, and dropping
    # the audio one would silently decide that the pruning cannot have moved
    # it, which is a claim nobody has measured.
    if torch.is_tensor(out):
        streams = {"video": out}
    elif isinstance(out, (list, tuple)) and out and all(
            torch.is_tensor(t) for t in out):
        names = ("video", "audio") if len(out) == 2 else \
            tuple(f"stream{i}" for i in range(len(out)))
        streams = dict(zip(names, out))
    else:
        print(f"[h3_capture] final tap: expected a tensor or a list of them, "
              f"got {type(out).__name__}; nothing written", flush=True)
        return

    # Same CPU-before-reshape discipline as the q/k/v path, for the same
    # reason: the model is near the card's limit at this moment.
    saved = {k: t.detach().cpu() for k, t in streams.items()}
    ref = saved["video"] if "video" in saved else next(iter(saved.values()))
    seq = ref.shape[2] if ref.ndim > 2 else ref.shape[-1]
    suffix = f"_r{render}" if render else ""
    name = (f"final_L{length_hint if length_hint is not None else 'na'}"
            f"_S{seq}_s{step}{suffix}.pt")
    path = os.path.join(_config["dir"], name)
    torch.save(saved, path)
    size = os.path.getsize(path) / 2**30
    shapes = " ".join(f"{k}{tuple(v.shape)}" for k, v in saved.items())
    print(f"[h3_capture] wrote {name}  {shapes} {ref.dtype}  {size:.3f} GiB", flush=True)


def summary():
    """What was seen, for confirming a capture run did what was asked."""
    if not enabled:
        return "capture disabled"
    if not _block_of:
        return ("[h3_capture] NO CALLS SEEN. The sage forward never ran. If "
                "Sol-Attn was on, it takes the calls inside its sigma window "
                "and sage keeps the rest -- 5 of 16 steps at the shipped "
                "0.2/0.9, which are steps 0-3 and 15. So Sol does NOT explain "
                "zero files; it explains a skewed sample. Check the block and "
                "step selectors first, then bypass Sol and re-run.")
    # `max()` on an empty dict raises, and `_calls` can now legitimately be
    # empty between renders. Read under the lock for the same reason.
    with _lock:
        blocks, steps, files = len(_block_of), max(_calls.values(), default=0), len(_written)
        renders = _render + 1
    declared = _config.get("cycle")
    note = "" if declared else (
        "  (no cycle= declared, so a second render in this process continues "
        "counting and will not capture)")
    return (f"[h3_capture] {blocks} blocks seen, {steps} steps, "
            f"{files} files written across {renders} render(s){note}")
