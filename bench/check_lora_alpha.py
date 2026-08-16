#!/usr/bin/env python3
"""Check every LoRA this repo names is one ComfyUI can scale correctly.

**The first check here whose SUBJECT is a third-party binary.** Two others come
close and neither is this: `check_prompt_guide_conformance.py` parses a file we
did not write, but grades *our* prompts with it, and `check_sol_kernel.py`
grades a dependency's *API*. Here the artifact under test is somebody else's
weights, because that is where the failure lives.

A LoRA's runtime scale is `alpha / rank`. ComfyUI learns `alpha` from one place
and one place only -- a `"<module>.alpha"` tensor (`comfy/lora.py:41-45`) --
and falls back to `1.0` when there is none
(`comfy/weight_adapter/lora.py:248-251`, and again on the bypass path at
`312-316`). **It never reads the file's `__metadata__`.**

diffusers' #14408 (2026-08-14) documents a published MiniMax-H3 turbo LoRA that
ships no `.alpha` tensors and records the alpha it trained with in
`__metadata__` under `alpha` instead: 8 against rank 128, an effective scale of
0.0625. A loader that assumes `alpha == rank` applies that adapter **16x too
strong**. Upstream fixed it on their side. On ours there is no fix available --
the loader has no channel for that metadata -- so the only defence is refusing
to ship a file of that shape, which is what this check is.

The failure is silent in every direction that matters: the graph validates, the
render completes, and the output is merely wrong. Nothing else here would
notice.

**The exemption is the hard part, and it is why this check is not three lines.**
"No `.alpha` keys and an alpha in the metadata" is NOT sufficient for guilt.
Three of the kijai `_resized_avg_` conversions in this install carry
`alpha: "8"` in `__metadata__` and are perfectly correct in ComfyUI, because
they also carry `baked_scale: "0.0625"` and their `conversion` string says the
scale is already folded into `lora_B`. Their intended runtime scale IS 1.0, so
ComfyUI's fallback is right and the metadata `alpha` is provenance, not an
instruction. A check that flagged them would cry wolf on three files a human
would then have to clear by hand, every run, and the wolf-crying is how a check
gets ignored on the day it is correct. So a declared bake wins over a declared
alpha, and the precedence is asserted by a control below rather than trusted.

Claims, i.e. what breaks if a case is deleted:

  resolves            every `*_LORA` constant in `h3_config.py` names a file
                      that exists. On 2026-08-16 `REF_LORA`'s symlink pointed
                      into an empty directory -- the file had moved one level
                      up -- so `h3_image_ref_plus_text_to_video_ref_lora.json`
                      could not run at all, and no check said so. Resolution
                      failure is reported apart from scale failure because the
                      two have nothing to do with each other.

  scale               no named LoRA hides a scale ComfyUI cannot see. This is
                      the 16x case.

  control:unsafe      the classifier actually rejects the shape #14408
                      describes. Synthesized here, so it runs on every
                      invocation: a check whose corpus is all-clean is
                      indistinguishable from a check that cannot fail, and
                      today every file in this install is clean.

  control:baked       the exemption still exists. Same synthetic file plus a
                      `baked_scale`, which must pass. It guards the opposite
                      direction from `control:unsafe`, and the split was
                      measured rather than reasoned: widening the exemption to
                      fire on the mere presence of an alpha key turns
                      `control:unsafe` red, so a widening is already covered;
                      removing the exemption turns only THIS case red. That
                      asymmetry matters because the files a narrowing would
                      wrongly condemn -- the three kijai `_resized_avg_`
                      conversions, verified flipping to "unsafe" without it --
                      are not in the graded set, so nothing else in this file
                      would notice until somebody named one in `h3_config.py`.

  premise             ComfyUI still reads alpha the way this check assumes.
                      Read from `comfy/lora.py`'s source, not imported. The
                      whole check is void the day the loader learns to read
                      `__metadata__`, and that day it should say so rather
                      than keep passing. Exits 2 if ComfyUI is not findable.

Shown red 2026-08-16, five mutations, each against a copy so the shipped file
was never edited:

  1. `REF_LORA` pointed at a name nothing resolves -- the real dangling-symlink
     state from that morning. `resolves` red, and the scale case for that
     constant correctly did NOT run: classifying nothing would have printed a
     green line for a file that does not exist.
  2. the hidden-alpha branch deleted -- `control:unsafe` red.
  3. the exemption widened to any *prose* mentioning alpha -- everything stayed
     green, correctly: no real or synthetic file changes verdict under it. A
     mutation that changes nothing is not evidence of anything, which is why
     3b exists.
  3b. the exemption widened to fire on the *presence* of an alpha key --
     `control:unsafe` red.
  4. a fake ComfyUI whose `comfy/lora.py` reads `__metadata__` -- `premise`
     red, naming the expiry.
  5. the exemption removed -- `control:baked` red, and the three kijai
     `_resized_avg_` conversions in this install classified "unsafe" while
     being correct.

Header-only: reads the JSON at the front of each safetensors file and never the
tensors. No CUDA, no model load, no server. About a second on a cold cache.
"""

from __future__ import annotations

import json
import re
import struct
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_COMFY = _REPO.parent.parent
sys.path.insert(0, str(_REPO / "workflows"))

import h3_config  # noqa: E402

failures = []
skipped = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# --------------------------------------------------------------------------
# Reading the header

def read_header(path):
    """The safetensors JSON header. Reads the front of the file, nothing else."""
    with open(path, "rb") as fh:
        (length,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(length))


def numeric(value):
    """A metadata value as a float, or None. Values are strings by spec."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


# A scale ComfyUI cannot see, if it is a number. `ss_network_alpha` is kohya
# provenance and is frequently the literal string "Dynamic", which is not a
# scale and must not be read as one -- hence `numeric()` rather than presence.
_ALPHA_KEYS = ("alpha", "network_alpha", "ss_network_alpha", "lora_alpha")

# A declaration that the scale is already in the weights, so 1.0 is correct.
# `baked_scale` is explicit. The prose keys are matched only for the verbs the
# producers in this install actually use, not for the word "alpha" -- see the
# control:baked case for why the pattern is kept this narrow.
_BAKED_KEYS = ("baked_scale", "application", "conversion", "training_scale")
_BAKED_PROSE = re.compile(r"\b(baked|folded|fold(?:s|ed)? into|pre-?scaled)\b", re.I)


def classify(header):
    """(verdict, reason) for one file's header. Verdict is ok / unsafe."""
    meta = header.get("__metadata__") or {}
    tensor_keys = [k for k in header if k != "__metadata__"]

    per_module = [k for k in tensor_keys if k.endswith(".alpha")]
    if per_module:
        # ComfyUI reads these directly and computes alpha / rank itself.
        return "ok", f"{len(per_module)} per-module .alpha tensors"

    baked = None
    for key in _BAKED_KEYS:
        value = meta.get(key)
        if value is None:
            continue
        if key == "baked_scale" or _BAKED_PROSE.search(str(value)):
            baked = f"{key}={value!r}"
            break

    hidden = None
    for key in _ALPHA_KEYS:
        if key in meta and numeric(meta[key]) is not None:
            hidden = (key, numeric(meta[key]))
            break

    if baked is not None:
        # A declared bake outranks a declared alpha: the metadata alpha is then
        # provenance, and the weights already carry the scale.
        return "ok", f"scale declared baked ({baked})"

    if hidden is not None:
        key, value = hidden
        ranks = sorted({
            header[k]["shape"][0] for k in tensor_keys
            if re.search(r"(lora_down|lora_A)\.weight$", k) and header[k].get("shape")
        })
        rank = ranks[0] if ranks else None
        ratio = f", ComfyUI would apply {rank / value:.3g}x too strong" if rank and value else ""
        return "unsafe", (
            f"__metadata__[{key!r}]={value:g} with no .alpha tensors and no "
            f"declared bake; rank {rank}{ratio}"
        )

    # Nothing hidden: ComfyUI's 1.0 is the only scale the file offers.
    return "ok", "no alpha anywhere; 1.0 is the file's only declared scale"


# --------------------------------------------------------------------------
# The LoRAs this repo names

def lora_constants():
    return sorted(n for n in dir(h3_config) if n.endswith("_LORA"))


def resolve(name):
    """Absolute path for a LoRA h3_config names, or None. Prefers ComfyUI's own
    resolver so `extra_model_paths.yaml` is honoured; falls back to the stock
    directory when ComfyUI is not importable."""
    try:
        sys.path.insert(0, str(_COMFY))
        import folder_paths  # noqa: WPS433
        found = folder_paths.get_full_path("loras", name)
        if found:
            return Path(found)
    except Exception:
        pass
    candidate = _COMFY / "models" / "loras" / name
    # `.exists()` follows symlinks, which is the point: a link into an empty
    # directory is exactly the 2026-08-16 failure and must not read as present.
    return candidate if candidate.exists() else None


print("every LoRA h3_config names resolves on disk:")
resolved = {}
for const in lora_constants():
    value = getattr(h3_config, const)
    path = resolve(value)
    resolved[const] = path
    check(f"{const} resolves", path is not None,
          str(path) if path else f"{value!r} -> nothing (dangling symlink, or not downloaded)")

print("\nno named LoRA hides a scale ComfyUI cannot see:")
for const, path in resolved.items():
    if path is None:
        continue  # already failed above; classifying nothing proves nothing
    verdict, reason = classify(read_header(path))
    check(f"{const} scale is visible", verdict == "ok", reason)


# --------------------------------------------------------------------------
# Controls. Every file in this install is currently clean, so without these the
# corpus cannot distinguish a working check from an inert one.

def synthetic(meta):
    """A minimal but real safetensors file: one rank-128 lora_A, given metadata."""
    tensors = {
        "diffusion_model.blocks.0.attn.out_proj.lora_A.weight": {
            "dtype": "F32", "shape": [128, 8], "data_offsets": [0, 128 * 8 * 4],
        },
        "__metadata__": meta,
    }
    blob = json.dumps(tensors).encode()
    handle = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    handle.write(struct.pack("<Q", len(blob)) + blob + b"\0" * (128 * 8 * 4))
    handle.close()
    return Path(handle.name)

print("\ncontrols:")
unsafe_file = synthetic({"alpha": "8"})
try:
    verdict, reason = classify(read_header(unsafe_file))
    check("control:unsafe -- #14408's shape is rejected", verdict == "unsafe", reason)
finally:
    unsafe_file.unlink()

# The same file plus the bake declaration the kijai conversions carry. If this
# one is also rejected the exemption is broken and three real files start
# failing; if `control:unsafe` passes while this fails, the check is crying
# wolf, which is the failure mode that gets a check ignored.
baked_file = synthetic({"alpha": "8", "baked_scale": "0.0625"})
try:
    verdict, reason = classify(read_header(baked_file))
    check("control:baked -- a declared bake is exempt", verdict == "ok", reason)
finally:
    baked_file.unlink()


# --------------------------------------------------------------------------
# The premise. This check is worthless if ComfyUI's loader ever grows a
# metadata channel, and it would keep passing quietly. Read the source.

print("\npremise -- ComfyUI still reads alpha only from .alpha tensors:")
loader = _COMFY / "comfy" / "lora.py"
adapter = _COMFY / "comfy" / "weight_adapter" / "lora.py"
if not loader.is_file() or not adapter.is_file():
    print(f"  SKIP  comfy/lora.py not found under {_COMFY}")
    skipped.append("premise")
else:
    source = loader.read_text()
    check("alpha comes from a '.alpha' key", '"{}.alpha".format' in source)
    check("the loader never reads __metadata__", "__metadata__" not in source,
          "a metadata channel now exists; this check's premise has expired")
    check("absent alpha still means scale 1.0", "alpha = 1.0" in adapter.read_text())


if failures:
    print(f"\n{len(failures)} failure(s): {', '.join(failures)}")
    sys.exit(1)
if skipped:
    # Exit 2, not 0. The scale cases are graded against an assumption about
    # ComfyUI's loader; if that assumption went unverified, a caller keying on
    # the exit code must be able to tell that apart from a clean pass.
    print(f"\n{len(skipped)} case(s) SKIPPED: {', '.join(skipped)}. "
          "Exit 2: the loader's behaviour was assumed, not read.")
    sys.exit(2)
print("\nall ok -- every named LoRA scales the way its producer intended")
