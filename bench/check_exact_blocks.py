#!/usr/bin/env python3
"""`MiniMaxH3ExactBlocks` rests on a name, so assert the thing the name stands for.

The node keeps its forward out of Sol-Attn's hands by setting
`_uses_optimized_attention` on it. Both of Sol's composition sites skip a
forward carrying that flag -- and that is a **branch on a private attribute of
somebody else's module**, which is the failure class CLAUDE.md names: an
assumption that has only ever met one implementation. If a vendored update
renames the flag or drops the guard, this node silently starts being composed,
its blocks quietly go back to sage, and nothing anywhere says so. The render
still succeeds. The number still prints.

So two things are checked, and neither of them needs a GPU, a model or a server:

1. **The contract still exists in the vendored source.** Both compose sites --
   `_apply_patch`'s patch-time loop and `_install_compose_hooks`'s run-time
   pre_hook -- must still read `_uses_optimized_attention`. Parsed from the
   AST rather than grepped, so a mention inside a comment or docstring does not
   satisfy it.

2. **The forward does what the module docstring says**, executably: it strips
   the attention override from what it passes down, and it does NOT mutate the
   caller's dict. The second half is the one worth having. `transformer_options`
   is the live dict the sampler threads through every block, so stripping in
   place would disable sage and Sol for the REST of the model -- a two-block
   request silently becoming a whole-model change, in the direction that looks
   like an improvement.

**What this does NOT establish:** that a real render routes a named block away
from both kernels. That needs the model, and the observable is the node's own
log line beside Sol's `keeping blocks [...] dense` -- see `docs/SOLATTN.md`.
This check covers the part that rots silently, not the part a render shows.

    python bench/check_exact_blocks.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOL_SRC = REPO / "vendor" / "sol_attn_minimax.py"
FLAG = "_uses_optimized_attention"

# The two functions in the vendored module that must keep skipping our forward.
# Named rather than "any function", so a guard moving out of one of them is a
# failure rather than being covered by the other.
COMPOSE_SITES = ("_apply_patch", "_install_compose_hooks")


def _functions(tree):
    """Every function in the module, including nested ones, by name."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, []).append(node)
    return out


def check_contract(problems):
    if not SOL_SRC.exists():
        problems.append(f"{SOL_SRC.relative_to(REPO)} is missing; the node's "
                        f"whole ordering argument is about that file")
        return
    tree = ast.parse(SOL_SRC.read_text())
    funcs = _functions(tree)
    before = len(problems)
    for site in COMPOSE_SITES:
        if site not in funcs:
            problems.append(
                f"vendor/sol_attn_minimax.py has no {site}(); the compose site "
                f"MiniMaxH3ExactBlocks documents was renamed or removed, so its "
                f"ordering claim needs re-deriving against the new shape")
            continue
        # The observable is a string constant equal to the flag used as a
        # VALUE -- `getattr(fwd, "_uses_optimized_attention", False)`.
        # Comments are not in the AST at all, so only docstrings could
        # masquerade.
        #
        # **Corrected 2026-08-29.** This skipped `ast.Expr` nodes whose value
        # is a Constant, with a comment claiming that excluded docstrings. It
        # did not: `ast.walk` is breadth-first over a queue, so the Expr's
        # Constant child is already enqueued by the time the Expr is
        # `continue`d, and a function whose body is exactly the bare string
        # `"_uses_optimized_attention"` satisfied the check. What actually
        # kept the red-proof honest was `node.value == FLAG` exact equality --
        # the prose control used a multi-line docstring, which is not equal to
        # the flag. Docstrings are now excluded by SUBTRACTING them, which is
        # the only way that works against a walk.
        docstring_constants = set()
        for fn in funcs[site]:
            for node in ast.walk(fn):
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    docstring_constants.add(id(node.value))
        found = False
        for fn in funcs[site]:
            for node in ast.walk(fn):
                if (isinstance(node, ast.Constant) and node.value == FLAG
                        and id(node) not in docstring_constants):
                    found = True
        if not found:
            problems.append(
                f"vendor/sol_attn_minimax.py::{site} no longer reads {FLAG!r}. "
                f"MiniMaxH3ExactBlocks relies on that skip to stay uncomposed; "
                f"without it, its blocks silently fall back to sage and the "
                f"render still succeeds. Re-derive the ordering before shipping.")
    if len(problems) == before:
        print(f"  ok    contract      both compose sites still read {FLAG!r}")


def check_forward(problems):
    # Order matters and getting it wrong is not subtle -- the same trap
    # `check_schema_defaults.py` documents. `custom_nodes/` has to be on the
    # path so the package imports by name, but the repo directory itself must
    # NOT be: it holds a `nodes.py`, and ComfyUI's own
    # `comfy_extras/nodes_minimax_h3` does a bare `import nodes` expecting
    # ComfyUI's. The ComfyUI root goes ahead of both.
    sys.path.insert(0, str(REPO.parent))          # custom_nodes/
    sys.path.insert(0, str(REPO.parents[1]))      # ComfyUI root, must win
    # Importing the node module pulls Comfy's model-management layer; this
    # check never executes a node and must not select the GPU.
    import comfy.cli_args
    comfy.cli_args.args.cpu = True

    pkg = REPO.name
    try:
        mod = __import__(f"{pkg}.exact_blocks", fromlist=["_exact_forward"])
    except Exception as exc:                       # pragma: no cover
        problems.append(f"cannot import exact_blocks: {exc!r}")
        return
    fwd = mod._exact_forward

    if not getattr(fwd, FLAG, False):
        problems.append(f"_exact_forward does not carry {FLAG}; Sol-Attn would "
                        f"compose it and the named blocks would run sage")

    # Stand in for ComfyUI's Attention.forward and record what it was handed.
    seen = {}

    class _Stub:
        @staticmethod
        def forward(self, x, rope_freqs=None, transformer_options=None):
            seen["options"] = transformer_options
            return "out"

    import types
    fake_comfy = types.ModuleType("comfy.ldm.minimax.model")
    fake_comfy.Attention = _Stub
    saved = sys.modules.get("comfy.ldm.minimax.model")
    sys.modules["comfy.ldm.minimax.model"] = fake_comfy
    try:
        caller_options = {"optimized_attention_override": "SENTINEL",
                          "sigmas": [1.0], "sol_block": 49}
        fwd(object(), "X", rope_freqs=None, transformer_options=caller_options)
    finally:
        if saved is None:
            del sys.modules["comfy.ldm.minimax.model"]
        else:
            sys.modules["comfy.ldm.minimax.model"] = saved

    passed = seen.get("options")
    if passed is None:
        problems.append("_exact_forward never reached Attention.forward")
        return
    if "optimized_attention_override" in passed:
        problems.append(
            "_exact_forward passed the attention override down, so a named "
            "block still reaches sage or Sol -- the node does nothing")
    else:
        print("  ok    strips        the override is gone from what it passes down")

    if "optimized_attention_override" not in caller_options:
        problems.append(
            "_exact_forward MUTATED the caller's transformer_options. That dict "
            "is threaded through every remaining block, so this would disable "
            "sage and Sol for the whole rest of the model while looking like a "
            "two-block change.")
    else:
        print("  ok    no mutation   the caller's transformer_options is intact")

    dropped = [k for k in ("sigmas", "sol_block") if k not in passed]
    for key in dropped:
        problems.append(f"_exact_forward dropped {key!r} on the way down; "
                        f"it must remove ONLY the override")
    if not dropped:
        print("  ok    passes rest   everything but the override is forwarded")


def main() -> int:
    problems = []
    print("MiniMaxH3ExactBlocks stays out of Sol's composition:")
    check_contract(problems)
    check_forward(problems)
    if problems:
        print(f"\n  FAIL  {len(problems)} problem(s):")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("\na named block reaches neither sage nor Sol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
