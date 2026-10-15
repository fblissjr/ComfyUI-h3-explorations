#!/usr/bin/env python3
"""Confirm the H3 chain composes and runs, after any node-pack update.

Verification comes from the log lines and the submitted graph, not the video.

    [h3] MiniMax H3 self-attention on sage (...)              graphs with a sage node
    node order: Sol's model input reaches the dense backend   Sol graphs, read from the graph
    [h3-sol] sparse (1, ..., 56, 128) tau=...                 Sol graphs only

Line 1 says the sage node patched the model; the default chain since
2026-09-15 has no sage node, so it is owed only by the sage arms. Line 3 says
sparse engaged at the configured tau. **Line 2 is the order check**: Sol is
installed over the dense backend's override, so a call Sol declines runs on that
backend and not on core's default. Until 2026-10-15 it was the log line
"[h3-sol] chaining onto an existing attention override", which the Sol node has
not printed since its redesign (e0b8cc55, 2026-09-27): it read MISSING on every
run after that, a red that could not go green. It is now read from the graph the
smoke submits (`graph_order`), and the chaining itself is held by
bench/check_sol_node_equivalence.py. Every log needle names the code literal that
prints it, held by bench/check_smoke_h3.py, so a needle the code stops printing
goes red there. At a length Sol declines (its own "dense (...): seq N < M" line),
"sparse ran" is said not owed and the run exits 2.
That seam is a protocol two third-party repos agree on and neither owns, so
it is worth re-checking on every update rather than assuming.

## The default length cannot exercise Sol, found 2026-09-08 by running it

`--length 39`, the default, renders a 12,264-token sequence. The shipped Sol
node pins `min_tokens` at 12288, so Sol declines every call and `sparse ran`
reads MISSING however many steps you ask for. That is not a stale needle and
not buffering: the needle fires correctly at `--length 49` (17,360 tokens) on
the same server and the same build, which is how this was established rather
than argued. `verbose` is not the cause either -- this file sets it True on
the node before submitting.

So a run at the default length says nothing about the Sol path in either
direction, and the Sol path is the one a kernel change puts at risk. Use
`--length 49` when the question is whether Sol still routes. The default is
left alone because it is the cheap "does the chain compose" run, and because
raising it silently would make every future reader think the shorter run had
been covering Sol all along.

## Two ways this file was wrong until 2026-08-14, both found by running it

**The sage needle never matched.** It looked for `"sage routing:"`, a string
that appears nowhere in this repo except in this file -- the docstring above
and the `WANT` list. `assert_chain.py` logs `"sage routed a {n}-token probe
on {kernel}"`. So the one line that is supposed to appear on *every* run
could not be found on any run, and nobody noticed because the default path
(no `--log`) returns 0 with a disclaimer and never evaluates it.

**And the two Sol lines were asserted unconditionally.** Sol-Attn shipped OFF
when this was written (the UI graphs carried the node bypassed and the API
graphs omitted it); it has been on by default since, per CLAUDE.md. On a graph
without the node those lines are *correctly* absent, and asserting them made a
fully compliant run report failure. `docs/SOLATTN.md` is the authority for
Sol's knobs. That is
the third-case trap CLAUDE.md names -- when something gains an "off" state,
every assertion about it inherits a new case, and "correctly absent" is not
"broken".

So the Sol lines are now gated on the submitted graph actually carrying a Sol
node, and a gated-out run exits 2 rather than 0: it did not verify the thing
its name implies, and must not read as though it did.

Two deliberate choices, both from getting them wrong first:

- **Enough steps to look like a render, and a count the grid accepts.** An
  earlier version used 4, which produces a smeared, incoherent clip
  indistinguishable from a failure -- so the artifact it leaves behind causes
  exactly the alarm it was meant to rule out. The default is now constrained
  from two directions rather than one: it must still be recognisably
  converging, and it must divide the PDD file's 32-point grid, because the
  PDD node owns the step count on the graphs that carry it and refuses a
  count that cannot tile. See `--steps` for which value satisfies both and
  why; **do not restate it here** -- this bullet said 10 for one commit after
  the default moved to 8, which is the drift this repo keeps paying for.
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
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

WF = Path(__file__).resolve().parent.parent / "workflows"
sys.path.insert(0, str(WF.parent))
from reference_order import VIDEO_SOURCE_CLASSES  # noqa: E402

# (label, needle, gate, source). The needles are matched against the ComfyUI log
# written during THIS run, so they must be the strings the code actually emits.
# `source` is (file, the literal in it that prints the line): bench/check_smoke_h3.py
# holds every needle to its source, so a needle the code no longer prints goes red
# there and not on the card. `gate` is what the submitted graph must carry for the
# line to be owed: "sage" a sage node, "sol" a Sol node.
#
# "node order" was a needle until 2026-10-15: "chaining onto an existing attention
# override", which the Sol node printed until its redesign (e0b8cc55, 2026-09-27)
# and no code has printed since, so the line could not go green. What it stood for,
# Sol installed over the dense backend's override, is a fact of the submitted graph
# and is read from it (`graph_order`); the chaining itself is held by
# bench/check_sol_node_equivalence.py (a call Sol declines runs on `previous`).
WANT = [
    ("sage engaged", "MiniMax H3 self-attention on sage", "sage", ("nodes.py", '"[h3] MiniMax H3 self-attention on sage')),
    ("sparse ran  ", "] sparse (", "sol", ("sol_attn_h3.py", 'f"sparse {tuple(qs.shape)}')),
]
# Sol's own line when it declines a call for its length, and the two literals that print it: at a length under its
# `min_tokens` it declines every call and "sparse ran" is not owed (the default length, "The default length cannot
# exercise Sol" above).
DECLINED = re.compile(r"\[h3-sol\] dense \((?:\d+, )*\d+\): seq (\d+) < (\d+)")
DECLINED_SOURCES = (("sol_attn_h3.py", 'f"dense {tuple(qs.shape)}: {reason}"'), ("sol_attn_h3.py", 'f"seq {q.shape[1]} < {min_tokens}"'))
#: The dense chain's nodes a Sol node is installed over: core's backend node (`h3_config.DENSE_BACKEND_NODE`) or sage.
BACKEND_NODE_IDS = ("ModelAttentionBackend", "MiniMaxH3SageAttention")


def graph_order(wf: dict) -> list[tuple[str, str | None]]:
    """For each Sol node of a graph, the dense node its model input reaches upstream, or None: (Sol's id, that class)."""
    out = []
    for nid, n in wf.items():
        if n["class_type"] not in SOL_NODE_IDS:
            continue
        at, found, seen = n["inputs"].get("model"), None, set()
        while isinstance(at, list) and str(at[0]) in wf and str(at[0]) not in seen:
            seen.add(str(at[0]))
            up = wf[str(at[0])]
            if up["class_type"] in BACKEND_NODE_IDS:
                found = f"{up['class_type']} {at[0]}"
                break
            at = up["inputs"].get("model")
        out.append((nid, found))
    return out
SAGE_NODE_IDS = ("MiniMaxH3SageAttention",)

# Every Sol node id a graph on this box can carry: the Triton pack's, the
# vendored CUDA one, and ours since 2026-08-30. A graph carrying any should
# have its Sol lines checked. **`MiniMaxH3SolAttn` was missing until
# 2026-08-31**, so this went quietly blind to every regenerated graph -- it
# found no Sol node and checked nothing, which reads exactly like a graph
# that wires no Sol.
# `MiniMaxH3Sol` since the redesign (2026-09-27); the older ids stay for older graphs.
SOL_NODE_IDS = ("SolAttnPatch", "SolAttnMiniMax", "MiniMaxH3SolAttn", "MiniMaxH3Sol")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--length", type=int, default=39)
    # 8, not 10, and the reason is the PDD node now owning the count on 11
    # graphs. 10 does not tile the 32-point grid, so `resolve_emit_steps`
    # RAISES -- which turned the DEFAULT smoke on any PDD graph into a dead
    # render the moment `--steps` stopped silently no-opping there. 8 divides
    # the grid AND divides `simple`'s 1,000-entry table, so it is exact on the
    # base graphs too, which 10 also was. An explicitly requested off-grid
    # count still raises: that is a caller asking for something that does not
    # exist, and inventing a nearby one would render what nobody asked for.
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--log", help="ComfyUI log file; if given, the lines are checked here")
    ap.add_argument("--workflow", default="h3_text_to_video_api.json",
                    help="API-format graph in workflows/ to submit. The default "
                         "carries the shipped chain (backend node + Sol), so "
                         "every line it owes is checked.")
    args = ap.parse_args()
    base = f"http://{args.host}"

    # A path with a separator is taken as-is, so a scratch graph can be smoked
    # without being written into workflows/, where the generator and the graph
    # checks would have opinions about it.
    wf_path = Path(args.workflow) if "/" in args.workflow else WF / args.workflow
    if not wf_path.is_file():
        print(f"no such workflow: {wf_path}")
        return 2
    wf = json.loads(wf_path.read_text())
    steps_applied_to: set[str] = set()
    for n in wf.values():
        ct = n["class_type"]
        if ct in ("MiniMaxH3ImageToVideo", "MiniMaxH3Conditioning"):
            n["inputs"]["length"] = args.length
            n["inputs"]["prompt"] = (
                "Live-action, cinematic. A woman in a dark coat walks along a "
                "rain-wet stone street past iron railings, the camera tracking "
                "with her.\n\nAudio: rain on stone, footsteps.")
        if ct == "MiniMaxH3Resolution":
            # Generated reference graphs wire conditioner geometry from this
            # node. Patching the downstream link would destroy the graph;
            # patch its owning widget instead.
            n["inputs"]["length"] = args.length
        if ct in VIDEO_SOURCE_CLASSES:
            # Do not decode hundreds of reference frames for a short smoke.
            # The typed compiler would cap them later, but the loader cost is
            # paid first and can hide the behaviour the smoke means to test.
            n["inputs"]["frame_load_cap"] = args.length
        # Two nodes can own the step count, and which one does is a property
        # of the graph. Since 0.83.0 a PDD graph carries no `BasicScheduler`
        # at all -- `MiniMaxH3PDDLoRA` emits SIGMAS and owns the count -- so
        # patching only the scheduler silently no-opped on every PDD graph
        # while the summary below went on printing the count that was asked
        # for. A smoke runner that misreports what it queued is worse than one
        # that cannot set the value.
        #
        # An off-grid request RAISES here rather than being snapped: the node
        # names the legal divisors, and inventing a nearby count would render
        # something the caller did not ask for. `--steps 1` is legal on the
        # shipped 32-point grid.
        if ct in ("BasicScheduler", "MiniMaxH3PDDLoRA"):
            n["inputs"]["steps"] = args.steps
            steps_applied_to.add(ct)
        if ct in SOL_NODE_IDS:
            n["inputs"]["verbose"] = True
        if ct == "SaveVideo":
            n["inputs"]["filename_prefix"] = "video/_smoketest"
        if ct == "VHS_VideoCombine":
            n["inputs"]["filename_prefix"] = "Video/_smoketest"

    # The log's size before submitting, so the needles are matched against what
    # THIS run wrote. Matching the whole file let a line from an earlier render
    # stand in for this one (CLAUDE.md: a log line can belong to someone else's
    # run); found 2026-09-15, when the default dropped sage and an earlier
    # render's sage line would still have matched.
    log_path = Path(args.log) if args.log else None
    log_start = log_path.stat().st_size if log_path and log_path.is_file() else 0

    pid = json.load(urllib.request.urlopen(urllib.request.Request(
        f"{base}/prompt", json.dumps({"prompt": wf, "client_id": str(uuid.uuid4())}).encode(),
        {"Content-Type": "application/json"}), timeout=60))["prompt_id"]
    # Report where the step count actually landed, not what was asked for.
    # An empty set means the graph owns its own count and this run is NOT the
    # step count on the command line -- say so rather than printing a number
    # that describes nothing that happened.
    if steps_applied_to:
        steps_note = f"{args.steps} steps (set on {'+'.join(sorted(steps_applied_to))})"
    else:
        steps_note = (f"steps NOT set -- no BasicScheduler or MiniMaxH3PDDLoRA "
                      f"in this graph; it ran its own count, not {args.steps}")
    print(f"submitted {pid[:8]}, {args.length} frames / {steps_note}", flush=True)

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
        print("prove the attention chain engaged -- a silent bypass also succeeds.")
        return 0

    # Does the graph we actually submitted carry a Sol node? If not, the two
    # Sol lines are correctly absent and asserting them would fail a compliant
    # run -- Sol ships OFF and API graphs omit it entirely.
    has_sol = any(n["class_type"] in SOL_NODE_IDS for n in wf.values())
    has_sage = any(n["class_type"] in SAGE_NODE_IDS for n in wf.values())
    owed = {"sol": has_sol, "sage": has_sage}

    if log_path is None or not log_path.is_file():
        print(f"\nrender succeeded, but --log does not exist: {log_path}")
        print("The attention-chain lines were NOT checked from a file. Read "
              "the live server terminal, or pass a path the launcher writes.")
        return 2
    with log_path.open("rb") as fh:
        # A log rotated or truncated mid-run is shorter than the offset; read
        # it whole rather than nothing.
        fh.seek(log_start if log_path.stat().st_size >= log_start else 0)
        text = fh.read().decode(errors="replace")
    missing, skipped = False, False
    # node order: read from the graph that was submitted, not from a log line
    if owed["sol"]:
        for sol, found in graph_order(wf):
            print(f"  node order    {'ok' if found else 'MISSING'}    Sol {sol} " + (f"over {found}" if found else "reads its model from no dense backend"))
            missing |= found is None
    declined = DECLINED.search(text)
    for label, needle, gate, _source in WANT:
        if not owed[gate]:
            if gate == "sol":
                print(f"  {label}  SKIP    no Sol node in the submitted graph")
                skipped = True
            else:
                print(f"  {label}  n/a     not owed by this graph's dense chain")
            continue
        ok = needle in text
        if not ok and needle == "] sparse (" and declined:
            print(f"  {label}  not owed at this length: Sol declined every call (seq {declined.group(1)} < {declined.group(2)}); "
                  "--length 49 runs it")
            skipped = True
            continue
        print(f"  {label}  {'ok' if ok else 'MISSING'}")
        missing |= not ok
    if missing:
        print("\nA missing line is not proof of breakage if the log is buffered.")
        print("Confirm the log is live (byte count growing) before concluding.")
        print("Then grep this repo for the needle: a string that appears only")
        print("in smoke_h3.py is a stale needle, which is how the sage line sat")
        print("unmatchable until 2026-08-14.")
        return 1
    if skipped and owed["sol"]:
        print("\nSol's sparse path was not exercised at this length. Exit 2, not 0: this run verified")
        print("the chain's order and the render, not that Sol routes. --length 49 checks it.")
        return 2
    if skipped:
        print("\nThe Sol lines were not checked, because the graph has no Sol")
        print("node, which every shipped video graph carries. Exit 2, not 0: this run")
        print("verified the dense chain and the render, not the composition seam.")
        print("Point --workflow at a Sol graph (h3_text_to_video_api.json) to check it.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
