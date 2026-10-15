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

No card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_smoke_h3.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, case, finish  # noqa: E402

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


case("every needle names the code that prints it, and it is there", needles_in_their_code)
case("each needle matches its line", needles_match_their_lines)
case("the node order, from the submitted graph", the_order_from_the_graph)

if __name__ == "__main__":
    sys.exit(finish())
