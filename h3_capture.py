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
    out = {"dir": None, "blocks": {0}, "steps": {1}, "cycle": None, "final": False}
    for part in spec.split(","):
        key, _, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if key == "dir":
            out["dir"] = os.path.expanduser(val)
        elif key == "cycle" and val:
            out["cycle"] = int(val)
        elif key == "final" and val:
            out["final"] = val.strip().lower() not in ("0", "false", "no", "off")
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
                print("[h3_capture] H3_CAPTURE set but no dir=; capture disabled")
            else:
                os.makedirs(_config["dir"], exist_ok=True)
                print(f"[h3_capture] ARMED: dir={_config['dir']} "
                      f"blocks={sorted(_config['blocks'])} steps={sorted(_config['steps'])}"
                      + (" final=on" if _config.get("final") else ""))
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
    with _lock:
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
        _calls[block] = step + 1
        if block not in _config["blocks"] or step not in _config["steps"]:
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

    record = {"q": qh, "k": kh, "v": vh,
              "kernel": kernel, "block": int(block), "step": int(step),
              "sigma": sigma, "seq_len": int(seq), "render": int(render)}
    if segments is not None:
        record["segments"] = segments

    # **The capture asserts its own shape before it is written.** Adopted from
    # the PDD lane, which hit two silent short-capture bugs in one day: a file
    # of the wrong length is not detectable later, because nothing downstream
    # knows what length it should have been. Cheap, and it fails at write time
    # where the cause is still on screen.
    for name, t in (("q", qh), ("k", kh), ("v", vh)):
        if t.shape != qh.shape or t.ndim != 4 or t.shape[2] != seq:
            raise RuntimeError(
                f"h3_capture: {name} is {tuple(t.shape)}, expected "
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
          f"{size:.2f} GiB")


def wants_final():
    """Whether `final=1` was declared. Read by the patching node."""
    _sync_spec()
    return bool(enabled and _config.get("final"))


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
              f"got {type(out).__name__}; nothing written")
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
    print(f"[h3_capture] wrote {name}  {shapes} {ref.dtype}  {size:.3f} GiB")


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
