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

## A second run of the same length on one server, found 2026-10-15

Sol prints each shape once (`sol_attn_h3._log_once`) until its node runs again,
and with the node cache on a repeated graph does not run it again. So the
second of two smokes at one length on one server printed neither its sparse
line nor its decline, and read "sparse ran MISSING" on a server that was
routing. When this run's part of the log lacks them, and the server served this run's
Sol node from its cache (`/history`), the line is read from earlier in the
log: after the last process start and the last time the Sol node ran
(`SINCE`), under a resolution line equal to this run's (`GEOMETRY`). The
smoke prints the line it read and its number. It is the line the cached node
logged for that geometry, not one this render printed; the resolution line
carries the canvas and length, not tau, which the printed line does. When the
sampler itself was served from the cache (two identical smokes back to back),
nothing was sampled and the run is not graded.

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
# A shape Sol has said once it does not say again: `sol_attn_h3._log_once` prints each key once, until the Sol node runs
# again (`reset_sol_attn_stats`, called when the node executes and prints its "MiniMaxH3Sol on" line) or the process
# starts again. With the node cache on, a second render of the same graph does not run the node again, so a smoke that
# repeats a length reads neither "sparse (" nor the decline in its own part of the log: found 2026-10-15, the second of
# two runs at 49 frames on one server read "sparse ran MISSING". The line that run would have printed is the one logged
# earlier, since the last of those two marks, for the same geometry: our resolution node prints its summary on every
# run, and it is how an earlier Sol line is tied to this run's length and canvas. (label, needle, (file, literal)).
# An earlier line is read only when the server served this run's Sol node from its cache (`/history`'s
# "execution_cached"): a node listed there was needed for this prompt's output and did not run, so it did not clear what
# Sol had said, and it is the node with the inputs it had when it last ran. A Sol node off the path to the output is not
# listed, and a node that ran prints its own "on" line. When the sampler itself is cached nothing was sampled, and the
# run is not graded. The sage line has the same gap (printed when the sage node runs, so not on a cached repeat) and is
# not read earlier: the default graph does not owe it.
SINCE = (("the process", "Total VRAM ", ("comfy/model_management.py", '"Total VRAM {:0.0f} MB, total RAM')),
         ("the Sol node", "] MiniMaxH3Sol on: ", ("sol_attn_h3.py", 'f"[h3-sol] MiniMaxH3Sol on: ')))
GEOMETRY = ("] resolution ", ("resolution.py", 'logger.info("[h3] resolution %s"'))
#: The nodes that sample: when the server served every one of them from its cache, this run sampled nothing.
SAMPLER_IDS = ("SamplerCustomAdvanced", "SamplerCustom", "KSampler", "KSamplerAdvanced")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _lines(text: str) -> list[str]:
    """The log's lines as the file numbers them: split on newlines only. tqdm's progress bar writes carriage returns
    inside a line, and `str.splitlines` would split on those too and count lines the file does not have."""
    return text.split("\n")


def _lines_with(text: str, needle: str) -> list[int]:
    return [i for i, line in enumerate(_lines(text)) if needle in line]


def _geometry(line: str) -> str:
    """What a resolution line says, from the needle on: a launcher's prefix (a time, a colour) is not compared."""
    return line[line.index(GEOMETRY[0]):].strip()


def logged_earlier(before: str, this_run: str, needle: str | re.Pattern) -> tuple[int, str] | None:
    """The line holding `needle` that Sol logged earlier for this run's geometry and has not been able to say since: in
    `before` (the log up to this run), after the last process start and the last time the Sol node ran, under a
    resolution line equal to this run's. (1-based line number in the file, the line), or None. None when this run ran
    the Sol node itself (its set is new) or printed no resolution line (nothing ties an earlier line to it)."""
    if any(_lines_with(this_run, n) for _l, n, _s in SINCE[1:]):
        return None
    geometry = next((_geometry(line) for line in _lines(this_run) if GEOMETRY[0] in line), None)
    if geometry is None:
        return None
    lines = _lines(before)
    # the geometry is read from the process's start; a line counts only after the Sol node last ran, which it does
    # inside a render, after that render's resolution line
    start = max([0] + _lines_with(before, SINCE[0][1]))
    ran = max([start] + [i for i in _lines_with(before, SINCE[1][1]) if i >= start])
    found, under = None, None
    for i in range(start, len(lines)):
        if GEOMETRY[0] in lines[i]:
            under = _geometry(lines[i]) == geometry
        elif under and i > ran and (needle.search(lines[i]) if isinstance(needle, re.Pattern) else needle in lines[i]):
            plain = _ANSI.sub("", lines[i]).strip()
            found = (i + 1, plain[plain.find("[h3-sol]"):] if "[h3-sol]" in plain else plain)
    return found


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
    whole = log_path.read_bytes()
    # A log rotated or truncated mid-run is shorter than the offset; read it whole rather than nothing.
    cut = log_start if len(whole) >= log_start else 0
    before, text = whole[:cut].decode(errors="replace"), whole[cut:].decode(errors="replace")
    cached = {str(n) for m in h.get(pid, {}).get("status", {}).get("messages", []) if m[0] == "execution_cached"
              for n in m[1].get("nodes", [])}
    lines, missing, skipped = grade(wf, text, before, cached)
    for line in lines:
        print(line)
    if any("NOT GRADED" in line for line in lines):
        return 2
    if missing:
        print("\nA missing line is not proof of breakage if the log is buffered, nor if it was said before:")
        print("Sol prints each shape once (sol_attn_h3._log_once) until the Sol node runs again or the server")
        print("restarts, and an earlier line is read only when this run's Sol node was served from the cache and")
        print("under a resolution line equal to this run's; the sage line prints only when the sage node runs, and")
        print("is never read earlier. Confirm the log is live (byte count growing) before concluding. Then grep")
        print("this repo for the needle: a string that appears only in smoke_h3.py is a stale needle, which is how")
        print("the sage line sat unmatchable until 2026-08-14.")
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


def grade(wf: dict, text: str, before: str = "", cached: set[str] = frozenset()) -> tuple[list[str], bool, bool]:
    """The lines the smoke prints for a submitted graph, the log this run wrote (`text`), the log before it (`before`)
    and the node ids the server served from its cache (`cached`), and whether a line is missing or a check was not
    graded."""
    owed = {"sol": any(n["class_type"] in SOL_NODE_IDS for n in wf.values()),
            "sage": any(n["class_type"] in SAGE_NODE_IDS for n in wf.values())}
    out, missing, skipped = [], False, False
    samplers = {nid for nid, n in wf.items() if n["class_type"] in SAMPLER_IDS}
    if samplers and samplers <= set(cached):
        return ([f"  sampled       NOT GRADED the server served the sampler ({', '.join(sorted(samplers))}) from its cache: "
                 "this run sampled nothing, so no attention ran. Change --length or restart the server"], False, True)
    sols = {nid for nid, n in wf.items() if n["class_type"] in SOL_NODE_IDS}
    # an earlier line stands for this run only when this run's Sol node was served from the cache (see SINCE)
    earlier = (lambda needle: logged_earlier(before, text, needle)) if sols and sols <= set(cached) else (lambda _n: None)  # noqa: E731
    # node order: read from the graph that was submitted, not from a log line
    if owed["sol"]:
        for sol, found in graph_order(wf):
            out.append(f"  node order    {'ok' if found else 'MISSING'}    Sol {sol} " + (f"over {found}" if found else "reads its model from no dense backend"))
            missing |= found is None
    declined, said = DECLINED.search(text), ""
    if declined is None and (again := earlier(DECLINED)):
        declined, said = DECLINED.search(again[1]), f", logged earlier by this server, line {again[0]}: {again[1]}"
    for label, needle, gate, _source in WANT:
        if not owed[gate]:
            if gate == "sol":
                out.append(f"  {label}  SKIP    no Sol node in the submitted graph")
                skipped = True
            else:
                out.append(f"  {label}  n/a     not owed by this graph's dense chain")
            continue
        ok = needle in text
        if not ok and needle == "] sparse (" and declined:
            out.append(f"  {label}  not owed at this length: Sol declined every call (seq {declined.group(1)} < {declined.group(2)}); "
                       f"--length 49 runs it{said}")
            skipped = True
            continue
        if not ok and needle == "] sparse (" and (again := earlier(needle)):
            out.append(f"  {label}  ok      logged earlier by this server for the same geometry, line {again[0]}: {again[1]} "
                       "(Sol says a shape once until its node runs again, sol_attn_h3._log_once; the server served "
                       "this run's Sol node from its cache)")
            continue
        out.append(f"  {label}  {'ok' if ok else 'MISSING'}")
        missing |= not ok
    return out, missing, skipped


if __name__ == "__main__":
    sys.exit(main())
