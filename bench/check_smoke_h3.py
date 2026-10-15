#!/usr/bin/env python3
"""The smoke's readings (`bench/smoke_h3.py`) held to the code they read and to the graphs it submits.

1. **Every log needle names the literal that prints it, and that literal is in its file.** The node-order needle was
   printed by the Sol node until its redesign (e0b8cc55, 2026-09-27) and by no code after, so from then the smoke read
   MISSING on every run, a red that could not go green, until 2026-10-15. A needle whose literal is gone goes red here.
2. **Each needle matches the line its literal prints**: the literal's template filled with a sample shape, prefixed the
   way the logger prefixes it, contains the needle; and Sol's decline line, made from its two literals, is read by
   `DECLINED` with its two numbers.
3. **The node order is read from the graph**: the shipped default graph's Sol reaches its dense backend; the same graph
   with Sol's model wired straight to the model loader reaches none (the control).
4. **A repeated length still reads what Sol said** (2026-10-15): a made log of one server where a 49-frame and a
   39-frame run each print their Sol line and a second run of each prints none (the Sol node served from the cache, so
   its seen set kept) grades the second 49 "ok" from the first's line and the second 39 "not owed" from the first's
   decline, each printing the line read and its number in the file (past tqdm's carriage returns); a launcher's prefix
   on the resolution line still matches; a run whose sampler was served from the cache is not graded. Controls, each
   red: the same repeat with the Sol node not cached, after it ran again, after a process start, under another
   geometry, and with no resolution line. The marks it reads (`SINCE`, `GEOMETRY`) are held to their code, core's
   process-start line in ComfyUI's own `comfy/model_management.py`.

No card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_smoke_h3.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import COMFY, REPO, case, finish  # noqa: E402

import smoke_h3 as sm  # noqa: E402


def needles_in_their_code() -> str:
    said = []
    for label, needle, _gate, (file, literal) in sm.WANT:
        text = (REPO / file).read_text(encoding="utf-8")
        assert literal in text, f"{label.strip()}: {file} no longer holds {literal!r}: the needle {needle!r} is stale"
        said.append(f"{label.strip()} in {file}")
    for file, literal in sm.DECLINED_SOURCES:
        assert literal in (REPO / file).read_text(encoding="utf-8"), f"Sol's decline line: {file} no longer holds {literal!r}"
    gone = "chaining onto an existing attention override"
    assert not any(gone in (REPO / f).read_text(encoding="utf-8") for f in ("sol_attn_h3.py", "attention.py", "nodes.py")), \
        "the old order line is printed again: the smoke could read it"
    return "; ".join(said) + "; Sol's decline from its two literals"


def needles_match_their_lines() -> str:
    shape = (1, 17360, 56, 128)
    sparse = "[h3-sol] " + f"sparse {tuple(shape)} tau=1.0 cuda-int8 rotate"          # `_log_once`'s prefix and the literal's head
    assert dict((w[0].strip(), w[1]) for w in sm.WANT)["sparse ran"] in sparse
    sage = "[h3] MiniMax H3 self-attention on sage (mode=auto, head_chunks=1, "
    assert dict((w[0].strip(), w[1]) for w in sm.WANT)["sage engaged"] in sage
    q_tokens, floor = 12264, 12288
    reason = f"seq {q_tokens} < {floor}"
    line = "[h3-sol] " + f"dense {(1, q_tokens, 56, 128)}: {reason}"
    got = sm.DECLINED.search("noise\n" + line + "\nmore")
    assert got and (got.group(1), got.group(2)) == (str(q_tokens), str(floor)), line
    assert sm.DECLINED.search("[h3-sol] dense (1, 12264, 56, 128): cross-attention (kept dense)") is None, "a decline for another reason read as length"
    return f"each needle is in the line its literal prints; the decline read as seq {q_tokens} < {floor}"


def the_order_from_the_graph() -> str:
    wf = json.loads((sm.WF / "h3_text_to_video_api.json").read_text())
    got = sm.graph_order(wf)
    assert got and all(found and found.startswith("ModelAttentionBackend") for _, found in got), got
    wrong = copy.deepcopy(wf)
    loader = next(nid for nid, n in wrong.items() if n["class_type"] == "UNETLoader")
    for nid, n in wrong.items():
        if n["class_type"] in sm.SOL_NODE_IDS:
            n["inputs"]["model"] = [loader, 0]
    assert all(found is None for _, found in sm.graph_order(wrong)), "the control: Sol over the bare model read as over a backend"
    return f"the shipped default graph's Sol is over {got[0][1]}; wired over the bare model it reaches none"


def a_repeat_reads_the_earlier_line() -> str:
    wf = json.loads((sm.WF / "h3_text_to_video_api.json").read_text())
    for label, needle, (file, literal) in sm.SINCE + (("the geometry",) + sm.GEOMETRY,):
        home = COMFY if file.startswith("comfy/") else REPO
        assert literal in (home / file).read_text(encoding="utf-8"), f"{label}: {file} no longer holds {literal!r}"
        assert needle in literal.replace("{:0.0f}", "1").replace('logger.info("', "").replace('f"', ""), (needle, literal)
    sol = {nid for nid, n in wf.items() if n["class_type"] in sm.SOL_NODE_IDS}
    sampler = {nid for nid, n in wf.items() if n["class_type"] in sm.SAMPLER_IDS}
    assert sol and sampler, (sol, sampler)
    cached = {nid for nid in wf if nid not in sampler and wf[nid]["class_type"] not in ("SaveVideo", "VHS_VideoCombine")}
    i = "\x1b[32m[INFO]\x1b[0m "
    start, on = i + "Total VRAM 24155 MB, total RAM 128563 MB", i + "[h3-sol] MiniMaxH3Sol on: sigma window [0, 0.9796], tau 1.0"
    g39 = i + "[h3] resolution 1344x768  7/4 | 39 frames (1.62s)  12 latent frames"
    g56 = i + "[h3] resolution 1344x768  7/4 | 56 frames (2.33s)  17 latent frames"
    sparse, dense = i + "[h3-sol] sparse (1, 17360, 56, 128) tau=1.0 cuda-int8 rotate", i + "[h3-sol] dense (1, 12264, 56, 128): seq 12264 < 12288"
    bar = "  0%|          | 0/8 [00:00<?, ?it/s]\r 12%|#         | 1/8 [00:03<00:21, 3.1s/it]"   # tqdm's carriage returns
    done = i + "Prompt executed in 26.99 seconds"
    before = "\n".join([start, "noise", g39, on, bar, dense, done, g56, bar, sparse, done]) + "\n"
    repeat = lambda g: "\n".join([g, bar, done]) + "\n"                              # noqa: E731 -- the Sol node cached: no Sol line
    lines, missing, skipped = sm.grade(wf, repeat(g56), before, cached)
    # THE CASE THAT GOES RED when a second run of a length on one server reads "sparse ran MISSING" again, or names
    # a line the file does not have there (the bar's carriage returns are not lines)
    assert not missing and not skipped and any("sparse ran" in x and " ok " in x and "line 10: [h3-sol] sparse (1, 17360" in x for x in lines), lines
    lines, missing, skipped = sm.grade(wf, repeat(g39), before, cached)
    assert not missing and skipped and any("not owed" in x and "12264 < 12288" in x and "line 6:" in x for x in lines), lines
    assert sm.grade(wf, "\n".join([g56, sparse, done]), before, cached)[1] is False, "this run's own line was not read"
    stamped = lambda line: "2026-10-15 20:06:00 " + line                              # noqa: E731 -- a launcher that stamps its lines
    assert not sm.grade(wf, repeat(stamped(g56)), before, cached)[1], "a prefix before the resolution line broke the match"
    lines, missing, skipped = sm.grade(wf, repeat(g56), before, cached | sampler)
    assert not missing and skipped and any("NOT GRADED" in x and "sampled nothing" in x for x in lines), lines
    # the controls: what the earlier line cannot stand in for
    for why, b, t, c in (("the Sol node was not served from the cache", before, repeat(g56), cached - sol),
                         ("the Sol node ran again before this run", before + on + "\n", repeat(g56), cached),
                         ("the Sol node ran in this run", before, "\n".join([g56, on, done]), cached - sol),
                         ("the server started again", before + start + "\n", repeat(g56), cached),
                         ("another geometry", before, repeat(i + "[h3] resolution 1344x768  7/4 | 73 frames (3.04s)  22 latent frames"), cached),
                         ("no resolution line in this run", before, done + "\n", cached)):
        lines, missing, _s = sm.grade(wf, t, b, c)
        assert missing and any("sparse ran" in x and "MISSING" in x for x in lines), (why, lines)
    return ("a second run at 49 reads the first's sparse line (file line 10, past a progress bar) and one at 39 the first's "
            "decline, each with the line; a stamped prefix still matches; a cached sampler is not graded; with the Sol node "
            "not cached, after it ran again, a restart, at another geometry or with no resolution line it reads MISSING")


case("every needle names the code that prints it, and it is there", needles_in_their_code)
case("each needle matches its line", needles_match_their_lines)
case("the node order, from the submitted graph", the_order_from_the_graph)
case("a repeated length reads what Sol said earlier", a_repeat_reads_the_earlier_line)

if __name__ == "__main__":
    sys.exit(finish())
