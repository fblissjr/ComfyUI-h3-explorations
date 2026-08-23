#!/usr/bin/env python3
"""Confirm the H3 chain composes and runs, after any node-pack update.

Verification comes from the log lines, not the video.

    [h3] ... sage routed a 2048-token probe on fp16_cuda      always
    [sol_attn] chaining onto an existing attention override   Sol graphs only
    [sol_attn] sparse (1, ..., 56, 128) tau=...               Sol graphs only

Line 1 says sage engaged. Line 3 says sparse engaged at the configured tau.
**Line 2 is the order check** -- it prints only when Sol-Attn finds sage's
override already installed. Missing on a Sol graph means the chain is
reversed and you are silently paying full price, with no error anywhere.
That seam is a protocol two third-party repos agree on and neither owns, so
it is worth re-checking on every update rather than assuming.

## Two ways this file was wrong until 2026-08-14, both found by running it

**The sage needle never matched.** It looked for `"sage routing:"`, a string
that appears nowhere in this repo except in this file -- the docstring above
and the `WANT` list. `assert_chain.py` logs `"sage routed a {n}-token probe
on {kernel}"`. So the one line that is supposed to appear on *every* run
could not be found on any run, and nobody noticed because the default path
(no `--log`) returns 0 with a disclaimer and never evaluates it.

**And the two Sol lines were asserted unconditionally.** Sol-Attn ships OFF --
derived from the graphs, not from a doc: every UI graph carries the node at
`mode=4` (bypass) and every API graph omits it. `docs/SOLATTN.md` is the
authority for Sol's knobs. This
smoke renders `h3_text_to_video_api.json`, so those lines are *correctly*
absent, and asserting them made a fully compliant run report failure. That is
the third-case trap CLAUDE.md names -- when something gains an "off" state,
every assertion about it inherits a new case, and "correctly absent" is not
"broken".

So the Sol lines are now gated on the submitted graph actually carrying a Sol
node, and a gated-out run exits 2 rather than 0: it did not verify the thing
its name implies, and must not read as though it did.

Two deliberate choices, both from getting them wrong first:

- **Enough steps to look like a render.** An earlier version used 4, which
  produces a smeared, incoherent clip indistinguishable from a failure --
  so the artifact it leaves behind causes exactly the alarm it was meant to
  rule out. 10 is still fast and still recognisably converging.
- **Its own filename prefix and a short clip.** The output is throwaway; it
  should not land in the middle of real renders wearing their naming.

Read the log in a terminal, or with `stdbuf -oL -eL` on the launcher.
Redirecting ComfyUI's output block-buffers it, and these lines then fail to
appear whether or not anything is wrong -- which makes an absent line
indistinguishable from a broken chain.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

WF = Path(__file__).resolve().parent.parent / "workflows"

# (label, needle, needs_sol). The needles are matched against the ComfyUI log,
# so they must be the strings the code actually emits -- `assert_chain.py`
# for the sage one. Anything here that no longer appears in the source is a
# stale needle, not a finding; grep before believing a MISSING.
WANT = [
    ("sage engaged", "sage routed a", False),
    ("node order  ", "chaining onto an existing attention override", True),
    ("sparse ran  ", "] sparse (", True),
]

# Both Sol node ids: the Triton pack's and the CUDA one. A graph carrying
# either should have its Sol lines checked.
SOL_NODE_IDS = ("SolAttnPatch", "SolAttnMiniMax")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--length", type=int, default=39)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--log", help="ComfyUI log file; if given, the lines are checked here")
    ap.add_argument("--workflow", default="h3_text_to_video_api.json",
                    help="API-format graph in workflows/ to submit. The default "
                         "ships Sol OFF, so the two Sol lines are skipped and "
                         "the run exits 2; pass a Sol graph to check the "
                         "composition seam.")
    args = ap.parse_args()
    base = f"http://{args.host}"

    # A path with a separator is taken as-is, so a scratch graph can be smoked
    # without being written into workflows/ where check_workflow_schema.py and
    # the generator would both have opinions about it.
    wf_path = Path(args.workflow) if "/" in args.workflow else WF / args.workflow
    if not wf_path.is_file():
        print(f"no such workflow: {wf_path}")
        return 2
    wf = json.loads(wf_path.read_text())
    for n in wf.values():
        ct = n["class_type"]
        if ct == "MiniMaxH3ImageToVideo":
            n["inputs"]["length"] = args.length
            n["inputs"]["prompt"] = (
                "Live-action, cinematic. A woman in a dark coat walks along a "
                "rain-wet stone street past iron railings, the camera tracking "
                "with her.\n\nAudio: rain on stone, footsteps.")
        if ct == "BasicScheduler":
            n["inputs"]["steps"] = args.steps
        if ct in SOL_NODE_IDS:
            n["inputs"]["verbose"] = True
        if ct == "SaveVideo":
            n["inputs"]["filename_prefix"] = "video/_smoketest"

    pid = json.load(urllib.request.urlopen(urllib.request.Request(
        f"{base}/prompt", json.dumps({"prompt": wf, "client_id": str(uuid.uuid4())}).encode(),
        {"Content-Type": "application/json"}), timeout=60))["prompt_id"]
    print(f"submitted {pid[:8]}, {args.length} frames / {args.steps} steps", flush=True)

    # /queue rather than the log: HTTP is never buffered.
    while True:
        q = json.load(urllib.request.urlopen(f"{base}/queue", timeout=10))
        if not q["queue_running"] and not q["queue_pending"]:
            break
        time.sleep(5)

    h = json.load(urllib.request.urlopen(f"{base}/history/{pid}", timeout=10))
    status = h.get(pid, {}).get("status", {}).get("status_str", "missing")
    print(f"render: {status}")
    if status != "success":
        return 1

    if not args.log:
        print("\npass --log <comfyui.log> to check the three composition lines,")
        print("or read them in the terminal. The render succeeding does not")
        print("prove sage or Sol-Attn engaged -- a silent bypass also succeeds.")
        return 0

    # Does the graph we actually submitted carry a Sol node? If not, the two
    # Sol lines are correctly absent and asserting them would fail a compliant
    # run -- Sol ships OFF and API graphs omit it entirely.
    has_sol = any(n["class_type"] in SOL_NODE_IDS for n in wf.values())

    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"\nrender succeeded, but --log does not exist: {log_path}")
        print("The attention-chain lines were NOT checked from a file. Read "
              "the live server terminal, or pass a path the launcher writes.")
        return 2
    text = log_path.read_text(errors="replace")
    missing, skipped = False, False
    for label, needle, needs_sol in WANT:
        if needs_sol and not has_sol:
            print(f"  {label}  SKIP    no Sol node in the submitted graph")
            skipped = True
            continue
        ok = needle in text
        print(f"  {label}  {'ok' if ok else 'MISSING'}")
        missing |= not ok
    if missing:
        print("\nA missing line is not proof of breakage if the log is buffered.")
        print("Confirm the log is live (byte count growing) before concluding.")
        print("Then grep this repo for the needle: a string that appears only")
        print("in smoke_h3.py is a stale needle, which is how the sage line sat")
        print("unmatchable until 2026-08-14.")
        return 1
    if skipped:
        print("\nThe Sol lines were not checked, because the graph has no Sol")
        print("node -- which is the shipped default. Exit 2, not 0: this run")
        print("verified sage and the render, not the composition seam.")
        print("Point --workflow at a Sol graph (h3_probe_sol_on.json) to check it.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
