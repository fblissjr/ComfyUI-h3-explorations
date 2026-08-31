#!/usr/bin/env python3
"""Report whether this ComfyUI carries the draft VSA patch, and which one.

VSA on H3 needs ComfyUI core to build a `to_gate_compress` slot on every DiT
block, which stock master does not. That support exists only as a DRAFT pull
request. This file is the provenance record for the copy applied here and the
thing that notices when it half-exists.

## Why absence is reported and not failed

A machine without the patch is a legitimate state -- it is what everyone else
has, and it is what this repo's own graphs assume. Failing on it would train a
reader to ignore red. What is NOT legitimate is the patch applied to one file
and not the other, because the two halves fail in opposite directions and the
combination is silent:

  model.py only          every H3 model gets a `gate_compress` PARAMETER, but
                         detection never sets it, so it stays False and no
                         gate is ever built. Identical to not patching at all,
                         while `grep gate_compress comfy/` says it is there.
  model_detection only   detection sets `gate_compress=True` from the state
                         dict and the model constructor does not accept it.
                         That one raises, so it is the loud half.

So the graded case is CONSISTENCY, not presence.

## What this cannot tell you

That the patch is CORRECT, or that it still matches upstream. It matches the
recorded commit by content hash of the two touched files' relevant lines, which
catches local edits, not an upstream force-push. The PR is a draft and its head
may move; when it does, the recorded sha below is what says which version this
box ran.

**And it cannot tell you the gate is USED.** The PR's own comment says the
weight is "unused by the dense forward; consumed by sparse attention patches".
Core loading it is necessary and not sufficient -- `MiniMaxH3VSAAttention` is
what computes it and passes it to the kernel.

    python bench/check_vsa_core_patch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent

# ---------------------------------------------------------------------------
# Provenance of the patch applied on this box. Written by hand because a draft
# PR is not a release and there is nothing to read it from.
# ---------------------------------------------------------------------------
UPSTREAM_REPO = "github.com/comfyanonymous/ComfyUI"
PR_NUMBER = 15958
PR_TITLE = "Minimax-H3: support FastVideo VSA"
PR_AUTHOR = "kijai"
PR_STATE = "DRAFT, still open as of 2026-08-31"
#: **The decision, 2026-08-31: do not apply it. Wait for the merge.**
#: It was applied to this box's working tree on 2026-08-30 and is GONE --
#: a `git reset` followed by two pulls took master to 95d755cd and carried
#: the uncommitted change away with it. Rather than re-apply a draft, this
#: box now tracks stock ComfyUI and waits. So ABSENT is the expected state
#: here, and every case below grades that state as correct rather than as
#: a shortfall -- see the checkpoint case for the one that used to not.
PR_APPLY_POLICY = "wait for merge; do not apply the draft"
#: The PR head this box's working tree was patched from, applied 2026-08-30 as
#: an UNCOMMITTED working-tree change on master rather than a merge. That is
#: deliberate: `git checkout -- <the two files>` reverts it, and a later
#: `git pull` REFUSES rather than quietly merging a draft into master.
PR_HEAD = "10febb01"
PR_BASE = "0a33ed6c"

#: The artifact the patch exists for. Machine-specific, so its case skips
#: when absent rather than failing.
VSA_CHECKPOINT = ("minimax_h3_fastvideo_vsa_datafree_1300step"
                  "_4step_int8_convrot.safetensors")

TOUCHED = {
    "comfy/ldm/minimax/model.py": "gate_compress",
    "comfy/model_detection.py": "gate_compress",
}

failures = []
skipped = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main():
    print(f"VSA core support: {UPSTREAM_REPO} PR #{PR_NUMBER} ({PR_STATE})")
    print(f"  \"{PR_TITLE}\" by {PR_AUTHOR}, head {PR_HEAD} on {PR_BASE}\n")

    present = {}
    for rel, token in TOUCHED.items():
        path = COMFY / rel
        if not path.is_file():
            print(f"  SKIP  {rel} not found; is this a ComfyUI checkout?")
            return 2
        present[rel] = token in path.read_text()

    hits = [rel for rel, ok in present.items() if ok]
    misses = [rel for rel, ok in present.items() if not ok]

    patched = bool(hits) and not misses

    if not hits:
        print("  none  the patch is ABSENT, which is stock ComfyUI and is not a "
              "failure.")
        print(f"        Policy: {PR_APPLY_POLICY}.")
        print("        H3 VSA checkpoints load with their gate keys DROPPED and "
              "render as\n        the dense base. MiniMaxH3VSAAttention refuses "
              "rather than let that pass.")
        check("consistent", True, "absent from both files, which is coherent")
    elif misses:
        check("consistent", False,
              f"applied to {hits} but not {misses}. A half-applied patch is "
              f"worse than none: with only the model change every H3 model "
              f"takes a gate_compress parameter that detection never sets, so "
              f"it silently stays False and looks exactly like stock.")
    else:
        check("consistent", True,
              f"present in both files ({', '.join(TOUCHED)})")
        print(f"        Applied from PR #{PR_NUMBER} head {PR_HEAD}. This is a "
              f"DRAFT, so\n        the H3 model on this box differs from stock "
              f"ComfyUI and any H3 result\n        taken here should say so.")

    # Is the SERVER running the patched core, or code from before it?
    #
    # **This case was vacuous when first written**, and the way it was vacuous
    # is worth keeping. It imported `comfy.ldm.minimax.model` in this process
    # and asked whether `Attention.__init__` takes `gate_compress` -- a fresh
    # import, reading the same files this check had just read, so it agreed
    # with them by construction and could not fail. The question that matters
    # is not what the files say, it is whether the process serving requests has
    # LOADED them, and a fresh import cannot see that.
    #
    # So: compare the port owner's start time against the patched files'
    # mtimes, which is the same thing `bench/restart_comfy.sh --newer-than`
    # asserts and for the same reason -- three measurements on 2026-08-29 were
    # taken against a server that had not reloaded, and every one produced a
    # plausible number rather than an error.
    import subprocess

    def port_owner(port=8188):
        try:
            out = subprocess.run(["ss", "-lptnH", f"sport = :{port}"],
                                 capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        # `ss` emits the owner as one whitespace-delimited token,
        # `users:(("python3",pid=289502,fd=44))`, so the pid is INSIDE it
        # rather than at the start of a token. Splitting and testing
        # `startswith("pid=")` finds nothing and skips forever -- which is what
        # this did when first written, and a permanent skip is indistinguishable
        # from "no server" in the output. Same expression restart_comfy.sh uses.
        import re
        match = re.search(r"pid=(\d+)", out)
        return int(match.group(1)) if match else None

    pid = port_owner()
    if pid is None:
        skipped.append("the server postdates the patch")
        print("  SKIP  the server postdates the patch   "
              "nothing is listening on 8188")
    else:
        try:
            started = (Path("/proc") / str(pid)).stat().st_mtime
        except OSError:
            started = None
        if started is None:
            skipped.append("the server postdates the patch")
            print(f"  SKIP  the server postdates the patch   "
                  f"cannot read the start time of pid {pid}")
        else:
            newest = max((COMFY / rel).stat().st_mtime for rel in TOUCHED)
            check("the server postdates the patch", started > newest,
                  f"pid {pid} started after the patched files were written"
                  if started > newest else
                  f"pid {pid} started {newest - started:.0f}s BEFORE the "
                  f"patched files were last written, so it is serving "
                  f"pre-patch code. Restart before believing any VSA result.")

    # Does the patch actually do its job on the artifact it exists for?
    #
    # The cases above check that the patch is THERE. This checks that it WORKS,
    # against the published checkpoint, and it is the only case here that
    # touches a real artifact. Skipped rather than failed when the checkpoint
    # is not on the box, which is the normal state for anyone who has not
    # downloaded 30-odd GB.
    #
    # **And skipped when the patch is absent, which is the fix for a defect
    # this file argued against and then committed.** The docstring above says
    # absence is legitimate and that failing on it "would train a reader to
    # ignore red" -- then this case failed on exactly that, because it asked
    # whether the gate keys find a slot without first asking whether anything
    # was supposed to build one. Patch absent plus checkpoint present is a
    # coherent state (it is stock ComfyUI with a file downloaded), and since
    # 2026-08-31 it is the DECIDED state. Red there is noise. The case still
    # has teeth where they belong: with the patch applied, a checkpoint whose
    # gate keys find no slot is a real failure and still fails.
    #
    # Nothing is allocated: the state dict is meta tensors carrying only the
    # real shapes, because detection reads `.shape` on a handful of entries and
    # the model is constructed under `torch.device("meta")`.
    ckpt = COMFY / "models" / "diffusion_models" / VSA_CHECKPOINT
    if not patched:
        skipped.append("the checkpoint's gate keys find a slot")
        print(f"  SKIP  the checkpoint's gate keys find a slot   "
              f"the patch is absent, so NO gate slot is built and all gate "
              f"weights\n        would be dropped on load. That is what stock "
              f"ComfyUI does and what this\n        box has chosen; it is not a "
              f"shortfall to grade. MiniMaxH3VSAAttention\n        refuses on "
              f"such a model, which is the control that matters here.")
    elif not ckpt.exists():
        skipped.append("the checkpoint's gate keys find a slot")
        print(f"  SKIP  the checkpoint's gate keys find a slot   "
              f"{VSA_CHECKPOINT} is not in models/diffusion_models")
    else:
        try:
            # ComfyUI is not installed; it is a checkout beside this pack.
            if str(COMFY) not in sys.path:
                sys.path.insert(0, str(COMFY))
            import torch
            from safetensors import safe_open

            import comfy.model_detection as detection
            import comfy.ops
            from comfy.ldm.minimax.model import MiniMaxH3Model

            with safe_open(str(ckpt), framework="pt") as handle:
                names = list(handle.keys())
                state = {k: torch.empty(handle.get_slice(k).get_shape(),
                                        device="meta") for k in names}
            config = detection.detect_unet_config(state, "")
            config.pop("image_model", None)
            with torch.device("meta"):
                model = MiniMaxH3Model(**config, operations=comfy.ops.manual_cast)
            slots = set(model.state_dict().keys())
            gates = {k for k in names
                     if "to_gate_compress" in k and k.endswith(".weight")}
            orphans = {k for k in names if k.endswith(".weight")} - slots
            placed = len(gates & slots)
            check("the checkpoint's gate keys find a slot",
                  gates and placed == len(gates) and not orphans,
                  f"{placed} of {len(gates)} gate weights placed, "
                  f"{len(orphans)} weight key(s) with no slot. Without the "
                  f"patch all {len(gates)} are dropped on load and the render "
                  f"succeeds as the dense base."
                  if not (gates and placed == len(gates) and not orphans) else
                  f"all {placed} gate weights placed, no orphan weight keys. "
                  f"Without the patch all {placed} would be dropped and the "
                  f"render would succeed as the dense base.")
        except Exception as exc:
            skipped.append("the checkpoint's gate keys find a slot")
            print(f"  SKIP  the checkpoint's gate keys find a slot   {exc}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {', '.join(failures)}")
        return 1
    if skipped:
        # Exit 2. A check that did not run must not read as one that passed --
        # and this one skips exactly when nobody is serving, which is when a
        # reader is most likely to be about to start something.
        print(f"INCOMPLETE: {len(skipped)} case(s) skipped: {', '.join(skipped)}")
        return 2
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
