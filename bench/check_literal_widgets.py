#!/usr/bin/env python3
"""A numeric widget must mean the quantity it names, not a mode.

## The rule

**A node input's value is the thing it is named after. A MODE is its own named
input -- a combo or a boolean -- never a magic value of a numeric one.**

`qwen_short_edge=0` does not mean "a short edge of zero pixels". It means "do
not give the text encoder a separate view at all", which is a different code
path reached by typing a number that looks like a size. The person setting it
has to know that 0 is not a size, and nothing on the widget says so.

## Why this is a check and not a style note

Two escaped instances, both already in CLAUDE.md, neither caught by anything:

  `nfe=0`             "the ordinary mode wearing a falsy sentinel" -- the
                      2026-08-28 audit's own words. 0 means "take the value
                      baked into the file", which is the NORMAL case.
  `qwen_short_edge=0` 0 means no separate encoder view. It is also the one
                      value CLAUDE.md says must not reach the shipped encoder,
                      and until 2026-08-31 it was what `execute()` defaulted to
                      -- so an API prompt omitting the key silently took the
                      path the docs forbid.

`check_schema_defaults.py` cannot see either: it compares a schema default
against a signature default and both were free to be the same sentinel.

## What this asserts, and what it cannot

It flags a declared numeric input whose own value is tested for TRUTHINESS or
compared to ZERO inside the module that declares it -- `if not x`, `bool(x)`,
`x == 0`, `if x:`. That is a proxy for "this number selects a branch", not a
proof of one, so the allowlist below carries the real judgement and is the part
to read.

**It cannot tell a sentinel from a legitimate guard.** `min_tokens` is compared
against a SEQUENCE LENGTH rather than against zero, so it does not trip this and
should not: 0 there means "gate nothing", which is literally what zero minimum
tokens means. That distinction -- compared to zero versus compared to another
quantity -- is the whole discrimination this check has.

So the allowlist is split by WHAT ZERO MEANS, three kinds:

  LITERAL_ZERO   zero means the quantity zero and the branch short-circuits
  SENTINELS      zero selects a MODE: the defect, carried as migration debt
  REFUSED_ZERO   zero is refused by a guard that raises, and the widget's own
                 minimum already excludes it, so no branch is selected at all.
                 The proxy trips on the comparison; the judgement is that a
                 refusal is not a mode (added 2026-09-05)

The point is not to relitigate the entries below. It is that a NEW one cannot
be added without someone writing a line here saying what the magic value means.

    python bench/check_literal_widgets.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Widgets whose value is compared against zero, split by WHAT ZERO MEANS.
#: The split is the point: nothing syntactic can tell these apart, so someone
#: has to say which kind a new one is, and that is the whole job of this check.

#: Zero means the quantity zero, and the branch is a short-circuit. Nothing
#: owed. These are here so the check does not have to guess, not as debt.
LITERAL_ZERO = {
    ("pdd_lora.py", "strength"): (
        "zero strength is zero strength. The `!= 0.0` guard skips computing "
        "un-merged blocks for a LoRA that would contribute nothing."),
    ("pdd_lora.py", "head_strength"): (
        "same: zero means the head patch contributes nothing, and the guard "
        "skips installing it."),
}

#: Zero selects a different MODE. This is the defect the check exists for, and
#: **migration is not free**: these are widget positions, and ComfyUI matches
#: saved widget values by INDEX, so converting one to a combo re-points every
#: later value in every saved graph and needs every graph regenerated -- the
#: same break the Sol node's v3 schema change took. Accepted debt, with the
#: replacement named so it is a decision rather than a shrug.
SENTINELS = {
    ("reference_conditioning.py", "qwen_short_edge"): (
        "**MIGRATED 2026-08-31 and kept here deliberately.** The widget is now "
        "`qwen_view`, a DynamicCombo of `separate` / `shared`, and the size "
        "lives under `separate` with min=CANVAS_MULTIPLE -- so 0 is no longer "
        "typeable and no longer selects anything. What still trips this check "
        "is `record.qwen_short_edge`, a DATACLASS FIELD on "
        "`RuntimeImageReference` where 0 remains the internal spelling for one "
        "shared view. That is a derived value nobody enters, which is outside "
        "what this rule governs -- but the check reads declared input NAMES "
        "and cannot see the difference, so the entry stays rather than the "
        "detector growing a special case it would be wrong about later."),
    ("reference_video_fit.py", "short_edge"): (
        "0 means REPORTING ONLY -- the node measures and warns and resizes "
        "nothing. Replacement: a boolean `resize` beside the size, or a "
        "combo `report only` / `downscale to N`."),
    ("pdd_lora.py", "nfe"): (
        "0 means take the evaluation count baked into the LoRA file, which is "
        "the ordinary case. Named by CLAUDE.md as a falsy sentinel on the "
        "ordinary mode. Replacement: a combo `from file` / `override (N)`."),
    ("pdd_lora.py", "steps"): (
        "0 means the file's own evaluation count -- `if not asked: return "
        "file_nfe`. Its default is 8, so unlike `nfe` the sentinel is not what "
        "you get by omission, but it is the same overload. Replacement: the "
        "same combo `nfe` needs, since the two knobs answer one question."),
    ("keyframe_canvas.py", "length"): (
        "`if length:` -- 0 means do not set a length at all rather than a "
        "length of zero frames. Replacement: leave the input optional and test "
        "for None, which already means absent without overloading a number."),
}

#: Zero is REFUSED: the comparison guards a `raise`, and the declared widget
#: cannot type zero in the first place. Nothing is selected, so nothing is owed;
#: the entry exists because the proxy cannot tell a refusal from a branch.
REFUSED_ZERO = {
    ("sol_chunked_h3.py", "chunk_rows"): (
        "zero is refused by a raising guard in `make_chunked_forward` "
        "(`chunk_rows % 64 or chunk_rows <= 0` raises ValueError) and the "
        "widget's min is 64, step 64, so no branch is selected: the widget "
        "means only rows per chunk. The author's reason for the guard: the "
        "producer requires 64-aligned chunk starts because its blocks are 64 "
        "rows. Red on this check from caa81ef (2026-09-01) until named here."),
}

ALLOWED = {**LITERAL_ZERO, **SENTINELS, **REFUSED_ZERO}

NUMERIC = {"Int", "Float"}


def declared_numeric_inputs(tree):
    """{input name: line} for every io.Int.Input / io.Float.Input in a module."""
    found = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Input"):
            continue
        base = node.func.value
        if not (isinstance(base, ast.Attribute) and base.attr in NUMERIC):
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            found[node.args[0].value] = node.lineno
    return found


def _direct_name(node):
    """The identifier this expression IS, or None if it is anything else.

    Direct only -- a bare `x` or an attribute chain ending `.x`. **Not a walk.**
    Walking was the first version and it was wrong in a way worth keeping: it
    recursed into call arguments, so `if not aspect_in_range(width, height)`
    read as a falsy test of `width` and flagged four widgets in three files
    that are plain dimensions. A check that fires on correct code is worse than
    no check, so the operand has to BE the knob, not merely contain it.
    """
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    return set()


def aliases(tree, names):
    """Locals assigned directly from a widget, e.g. `forced_nfe = int(nfe)`.

    One level, deliberately. `nfe` is renamed to `forced_nfe` before anything
    tests it, so a purely name-keyed search finds nothing and the widget looks
    clean -- which is how the check first reported CLAUDE.md's own worked
    example of a falsy sentinel as absent. Chasing further would need real
    dataflow; one hop covers the rename-then-test idiom that actually occurs
    and stops short of pretending to more.
    """
    out = dict.fromkeys(names)
    out = {n: {n} for n in names}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        # A pure RENAME or CAST only: the right-hand side must read exactly
        # one identifier, and that one is the widget. `forced_nfe = int(nfe)`
        # qualifies; `w, h = adapt_canvas(width, height)` and
        # `scale = short_edge / min(w, h)` do not.
        #
        # **The looser version -- any widget appearing anywhere on the right --
        # produced four false positives immediately**, because a dimension fed
        # into a helper made the helper's result an alias of the dimension, and
        # any later zero-test on that result flagged the widget. Same failure as
        # the walking operand above, one level up.
        called = {sub.func.id for sub in ast.walk(node.value)
                  if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)}
        read = {sub.id for sub in ast.walk(node.value)
                if isinstance(sub, ast.Name)} - called
        if len(read) == 1:
            (only,) = read
            if only in names:
                out[only].add(target.id)
    return out


def zero_tests(tree):
    """{name: [lines]} where a name is tested falsy or compared against 0."""
    hits: dict[str, list[int]] = {}

    def record(names, lineno):
        for n in names:
            hits.setdefault(n, []).append(lineno)

    for node in ast.walk(tree):
        # `x == 0`, `x > 0`, `x != 0` -- a comparison against the literal zero.
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and comparator.value == 0:
                    record(_direct_name(node.left), node.lineno)
        # `not x`
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            record(_direct_name(node.operand), node.lineno)
        # `bool(x)`
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "bool"):
            for arg in node.args:
                record(_direct_name(arg), node.lineno)
        # `if x:` / `elif x:` where x is the bare knob or an attribute of one
        elif isinstance(node, ast.If) and isinstance(node.test, (ast.Name, ast.Attribute)):
            record(_direct_name(node.test), node.lineno)
        # `x or fallback` -- the fallback-sentinel idiom, and the one that
        # reads most like ordinary code. Only the LEFT operand counts: being
        # somebody else's fallback says nothing about your own zero.
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            if node.values:
                record(_direct_name(node.values[0]), node.lineno)
    return hits


def main() -> int:
    print("numeric widgets must name a quantity, not select a mode\n")
    findings = []
    for path in sorted(REPO.glob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        declared = declared_numeric_inputs(tree)
        if not declared:
            continue
        tested = zero_tests(tree)
        alias_map = aliases(tree, declared)
        for name, decl_line in sorted(declared.items()):
            lines = sorted({ln for alias in alias_map[name]
                            for ln in tested.get(alias, ())})
            if lines:
                findings.append((path.name, name, decl_line, lines))

    unexplained = [f for f in findings if (f[0], f[1]) not in ALLOWED]
    accounted = [f for f in findings if (f[0], f[1]) in ALLOWED]

    for fname, name, line, lines in accounted:
        kind = ("literal" if (fname, name) in LITERAL_ZERO else
                "refused" if (fname, name) in REFUSED_ZERO else "SENTINEL")
        print(f"  ok   {fname}::{name}  [{kind}]  (declared line {line}, "
              f"branches {lines})")
        print(f"       {ALLOWED[(fname, name)]}")

    stale = [k for k in ALLOWED if k not in {(f[0], f[1]) for f in findings}]
    for key in sorted(stale):
        # An allowlist entry whose sentinel is gone is a lie about the code,
        # and it is the entry a reader trusts most because nobody removes them.
        print(f"  FAIL {key[0]}::{key[1]} is allowlisted but no longer tests "
              f"its value against zero. If the sentinel was removed, delete "
              f"the entry.")

    for fname, name, line, lines in unexplained:
        print(f"  FAIL {fname}::{name}  declared line {line}, its own value is "
              f"tested against zero at {lines}.")
        print(f"       A numeric widget must mean the quantity it names. Make "
              f"the mode its own input -- a combo or a boolean -- or, if this "
              f"really is a bound rather than a mode, add it to ALLOWED with "
              f"what the value means.")

    print()
    if unexplained or stale:
        print(f"FAILED: {len(unexplained) + len(stale)} widget(s) select a mode "
              f"by numeric value without saying so.")
        return 1
    n_sent = sum(1 for f in accounted if (f[0], f[1]) in SENTINELS)
    print(f"all {len(accounted)} zero-compared widget(s) accounted for "
          f"({n_sent} sentinel(s) carrying migration debt); no new ones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
