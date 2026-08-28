#!/usr/bin/env python3
"""Check the generator reads upstream constants rather than repeating them.

Sharing a constant and asserting it is shared are different guarantees with
different timelines. Sharing prevents drift: nobody can edit one of two copies
because there is only one. This check prevents un-sharing: someone writing
`2048` back into a builder six months from now because the literal reads more
directly at the call site, or copying a call into a third place without
wiring it to the shared source.

Sharing it once is not a vaccination. This repo has the receipt: the
generator wrote `short_edge: 2048` as a literal on 2026-08-11 in the same
session that argued for reading constants rather than repeating them.

**Agreement is not a testable property.** Asserting the graph matches
upstream's value passes identically whether the generator reads the constant
or hardcodes today's copy of it, because they agree until the day they do
not. So each case here forces a disagreement -- it moves upstream's value and
asserts the graph follows -- which only holds if the generator genuinely
reads it.

Needs ComfyUI importable. Builds graphs in memory; touches no server and
writes no files.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "workflows"))
sys.path.insert(0, str(_REPO.parent.parent))  # ComfyUI root

# Import order matters. `build_workflows` puts this repo's root on sys.path
# for its own `h3_rules` import, and our `nodes.py` then shadows ComfyUI's,
# which `comfy_extras.nodes_minimax_h3` imports by that name. Pull core in
# first, while ComfyUI's root is still the one that answers.
import comfy_extras.nodes_minimax_h3 as core  # noqa: E402
import build_workflows as bw  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def ref_short_edge_in(graph):
    """The short_edge every image append in an API graph was given.

    Read from `MiniMaxH3AppendRefImage` since the fit fold, which is where the
    knob lives now. Reading it from the retired `MiniMaxH3ReferenceFit` did not
    fail loudly when the graphs stopped carrying that node: the set comprehension
    simply went empty, and an empty set compares unequal to every value, so the
    three cases below went red for the right reason by luck rather than by
    design. An empty set is asserted against explicitly now.

    **Both spellings, dotted first.** `size_policy` became a DynamicCombo on
    2026-08-27 (`e6e527e`), so the API form spells its members
    `size_policy.short_edge`, never the flat `short_edge`. This reader was left
    on the flat name and every graph went subject-less, which raised the
    "lost its subject" assertion below on a subject that had only been renamed.
    `bench/preflight_graph.py` took the same fix in `d7dd575`; the flat name is
    still tried so a hand-built graph on the old spelling is still priced.
    """
    found = set()
    for n in graph.values():
        if n["class_type"] != "MiniMaxH3AppendRefImage":
            continue
        for candidate in ("size_policy.short_edge", "short_edge"):
            if candidate in n["inputs"]:
                found.add(n["inputs"][candidate])
                break
    if not found:
        raise AssertionError(
            "no MiniMaxH3AppendRefImage in this graph carries short_edge; this "
            "check has lost its subject and would otherwise report a "
            "difference it never measured")
    return found


print("reference short edge follows comfy_extras.nodes_minimax_h3:")
original = core.REF_IMAGE_SHORT_EDGE
try:
    before = ref_short_edge_in(bw.build_api("r2v"))
    check("the graph carries upstream's value", before == {original},
          f"graph {before}, upstream {original}")

    # Force the disagreement. A generator that hardcodes stays at `original`
    # here, and that is the only observable difference between the two.
    setattr(core, "REF_IMAGE_SHORT_EDGE", original + 512)
    moved = ref_short_edge_in(bw.build_api("r2v"))
    check("it follows when upstream moves", moved == {original + 512},
          f"graph {moved}, upstream {original + 512}")
finally:
    setattr(core, "REF_IMAGE_SHORT_EDGE", original)

restored = ref_short_edge_in(bw.build_api("r2v"))
check("and follows back", restored == {original}, f"graph {restored}")

if failures:
    print(f"\n{len(failures)} failed: {', '.join(failures)}")
    raise SystemExit(1)
print("\nthe generator reads its constants")
