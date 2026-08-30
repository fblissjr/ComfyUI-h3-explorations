"""Parsing for the block-index specs several nodes take as a string widget.

One function, and it lives alone because four callers had reached into a NODE
module to get it: `exact_blocks.py` imported it from the vendored Sol node,
`pdd_lora.py` from ours with a three-way fallback, `bench/time_dense_blocks.py`
through `importlib` with a `fromlist`, and `sol_attn_h3.py` defined it. That
last one is why the others had to reach: a shared parser was living inside the
Sol-Attn node because that is where it happened to be written first, so
depending on it meant depending on Sol-Attn.

**The reason to move it is the vendored node's retirement, not tidiness.**
`vendor/sol_attn_minimax.py` becomes a read-only reference, and a reference
copy that live code imports from is not a reference -- it is a dependency that
nobody may fix. Two callers were importing from it.

`bench/build_hybrid.py` keeps its own `parse_blocks` and is NOT a fifth caller.
It takes no `count`, so it cannot resolve a negative index and does not clamp;
its specs name blocks of a checkpoint being assembled rather than of a loaded
model. Merging them would mean giving that one a count it has no way to know.
Two functions with one name is a smell, so it is written down here rather than
left for the next reader to spot and "fix".
"""

from __future__ import annotations

import re


def parse_blocks(spec, count):
    """Parse "0-3,47,-1" into absolute block indices; negatives count from the end.

    `count` is the number of blocks in the model, which is what makes a
    negative index resolvable and what the result is clamped to. Ranges may be
    given either way round. Whitespace anywhere is ignored, so a spec pasted
    across lines still parses.

    Byte-identical to the definition it replaces, in the vendored Sol node and
    in `sol_attn_h3.py` -- verified by diff before the move, because a silent
    behaviour change in a parser that decides WHICH BLOCKS get different
    treatment would show up as a quality difference, not as an error.
    """
    out = set()
    for part in "".join(str(spec).split()).split(","):   # tolerate any whitespace
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part)
        if match is None:
            raise ValueError(f"cannot parse block spec {part!r}; "
                             "use indices and ranges like '0-3,47,-1'")
        first = int(match.group(1))
        last = first if match.group(2) is None else int(match.group(2))
        first = first if first >= 0 else count + first
        last = last if last >= 0 else count + last
        if first > last:
            first, last = last, first
        out.update(range(max(first, 0), min(last, count - 1) + 1))
    return frozenset(out)
