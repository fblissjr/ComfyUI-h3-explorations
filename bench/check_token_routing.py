#!/usr/bin/env python3
"""`token_routing` on MiniMaxH3SolAttn fills `token_aug_blocks`; assert what it fills.

The dropdown exists so nobody has to remember `blocks=budget` syntax, which
makes its failure mode a quiet one: a preset that resolves to the wrong blocks
still renders, and the render still looks like a render. Three things would be
wrong without anything else noticing, and each is asserted here:

1. **A saved graph changes meaning.** Every graph written before the widget
   existed carries no value for it, so the default has to reproduce
   `parse_token_aug_profile` on the text exactly, empty text included.
2. **A preset reaches the last blocks unguarded.** Token routing RAISED the
   error on block 49 unless `qk_balance` and `rotate` were both on
   (`bench/results/2026-09-15_sol_token_aug_x_options_b49_s15.json`), so the
   "early and middle" preset must stop short of the last five and "all
   blocks" must be refused without the two options.
3. **Two widgets disagree.** A preset with text also typed has no right
   answer, so it is refused; "off" is the one mode that may ignore text.

Plus the widget's position: `bench/check_node_ids.py` matches widget values by
index, so it has to be declared last. No GPU, no model, no server.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
H3_BLOCKS = 50


def _load():
    # Same path order as check_exact_blocks.py, for the reason given there.
    sys.path.insert(0, str(REPO.parent))
    sys.path.insert(0, str(REPO.parents[1]))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    return __import__(f"{REPO.name}.sol_attn_h3", fromlist=["resolve_token_routing"])


def _raises(problems, label, fn):
    try:
        got = fn()
    except ValueError:
        return
    problems.append(f"{label}: expected a refusal, got {got!r}")


def main() -> int:
    problems: list[str] = []
    print("token_routing resolves to the blocks its label names:")
    try:
        m = _load()
    except Exception as exc:                       # pragma: no cover
        print(f"  FAIL  cannot import sol_attn_h3: {exc!r}")
        return 1
    r = m.resolve_token_routing
    n = H3_BLOCKS

    # 1. off is the default and means off; custom reads the list; a pre-widget
    #    graph (mode None) keeps the meaning it had: the list decides
    for spec in ("", "0,24,32=64"):
        if r(m.TOKEN_ROUTING_OFF, spec, n) != {}:
            problems.append(f"'off' did not resolve to off with list {spec!r}")
    for spec in ("0,24,32=64", "0-10=128; 40=64"):
        want = m.parse_token_aug_profile(spec, n)
        if r(m.TOKEN_ROUTING_CUSTOM, spec, n) != want:
            problems.append(f"'custom' does not read the list {spec!r}")
        if r(None, spec, n) != want:
            problems.append(f"a pre-widget graph (mode None) lost its list {spec!r}")
    if r(None, "", n) != {}:
        problems.append("a pre-widget graph with no list did not resolve to off")
    _raises(problems, "custom with an empty list", lambda: r(m.TOKEN_ROUTING_CUSTOM, "", n))
    _raises(problems, "the retired 'text field' value", lambda: r("text field", "", n))

    # 2. where each preset reaches
    got = r(m.TOKEN_ROUTING_MEASURED, "", n)
    if got != {b: m.TOKEN_ROUTING_BUDGET for b in (0, 24, 32, 40)}:
        problems.append(f"measured preset resolved to {sorted(got)}")
    got = r(m.TOKEN_ROUTING_EARLY_MIDDLE, "", n)
    if sorted(got) != list(range(n - 5)) or set(got.values()) != {m.TOKEN_ROUTING_BUDGET}:
        problems.append(f"early-and-middle preset resolved to {min(got)}..{max(got)}, "
                        f"expected 0..{n - 6}")
    for qb, rot in ((False, False), (True, False), (False, True)):
        _raises(problems, f"all blocks with qk_balance={qb} rotate={rot}",
                lambda qb=qb, rot=rot: r(m.TOKEN_ROUTING_ALL, "", n, qk_balance=qb, rotate=rot))
    got = r(m.TOKEN_ROUTING_ALL, "", n, qk_balance=True, rotate=True)
    if sorted(got) != list(range(n)):
        problems.append(f"all-blocks preset resolved to {len(got)} blocks of {n}")
    _raises(problems, "measured preset on a 10-block model",
            lambda: r(m.TOKEN_ROUTING_MEASURED, "", 10))

    # 3. a preset and typed text together
    for mode in (m.TOKEN_ROUTING_MEASURED, m.TOKEN_ROUTING_EARLY_MIDDLE):
        _raises(problems, f"{mode!r} with text typed", lambda mode=mode: r(mode, "0=64", n))
    _raises(problems, "an unknown mode", lambda: r("everything", "", n))

    # the widget: last, optional, defaulting to off
    schema = m.MiniMaxH3SolAttn.define_schema()
    last = schema.inputs[-1]
    if last.id != "token_routing":
        problems.append(f"token_routing is not the last declared input (last is {last.id!r}); "
                        "saved graphs match widget values by index")
    elif last.default != m.TOKEN_ROUTING_OFF or list(last.options) != list(m.TOKEN_ROUTING_MODES):
        problems.append("token_routing's default or options differ from the module's table")

    if problems:
        print(f"\n  FAIL  {len(problems)} problem(s):")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("\nsaved graphs keep their meaning; no preset reaches the last blocks unguarded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
