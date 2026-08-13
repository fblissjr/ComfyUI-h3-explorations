#!/usr/bin/env python3
"""Check the reference short-edge override applies once and never leaks.

The override rebinds a module constant that ComfyUI's own node reads at call
time. That is a process-global mutation, which is the exact shape of the
contamination this repo guards against elsewhere: an override that outlived
its call would change references in graphs that never asked for it, silently,
and only for renders queued after the one that armed it.

So the properties worth pinning are about scope, not about arithmetic:

  applies_once            an armed override reaches the next call and the
                          call after it runs stock. Delete and a leak looks
                          identical to correct behaviour on the first render
  restores_on_raise       the constant goes back even when the wrapped call
                          throws. Delete and one failed render poisons the
                          session
  inert_under_match       with ref_image_size on 'match' the stock node never
                          reads the constant, so the override must decline and
                          say so rather than appear to work
  install_is_idempotent   wrapping twice would apply the override twice and
                          consume one arm per layer. The chaining packs use a
                          marker for this; so do we

Driven through `_make_wrapper` with a stub rather than the real node, so it
needs no VAE, no CLIP and no model. Needs comfy importable for the constant.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO.parent))

rf = importlib.import_module(f"{_REPO.name}.reference_fit")
import comfy_extras.nodes_minimax_h3 as core  # noqa: E402

BASE = core.REF_IMAGE_SHORT_EDGE
failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def seen_constant(**kwargs):
    """Stand-in for the stock execute: reports the constant it would read."""
    return core.REF_IMAGE_SHORT_EDGE


print("applies_once:")
wrapped = rf._make_wrapper(seen_constant)
rf.arm_short_edge_override(3072)
first = wrapped(ref_image_size="max")
second = wrapped(ref_image_size="max")
check("armed call sees the override", first == 3072, f"saw {first}")
check("the next call sees the default", second == BASE, f"saw {second}")
check("constant restored after the call", core.REF_IMAGE_SHORT_EDGE == BASE,
      f"is {core.REF_IMAGE_SHORT_EDGE}")

print("\nrestores_on_raise:")


def boom(**kwargs):
    raise RuntimeError("kernel went bang")


wrapped_boom = rf._make_wrapper(boom)
rf.arm_short_edge_override(3072)
try:
    wrapped_boom(ref_image_size="max")
except RuntimeError:
    pass
check("constant restored after a raise", core.REF_IMAGE_SHORT_EDGE == BASE,
      f"is {core.REF_IMAGE_SHORT_EDGE}")

print("\ninert_under_match:")
rf.arm_short_edge_override(3072)
got = wrapped(ref_image_size="match")
check("declines under match", got == BASE, f"saw {got}")
check("and consumes the arm rather than holding it",
      wrapped(ref_image_size="max") == BASE)

print("\nper_node_arms:")
# Two fit nodes in one graph, one armed and one not. With a single global
# value the unarmed one's disarm wiped the armed one's entry, and which won
# came down to ComfyUI's execution order between independent nodes -- not the
# graph's visual order, and not settable. Same graph, two possible renders.
rf.disarm_short_edge_override()
rf.arm_short_edge_override(3072, node_id="11")
rf.disarm_short_edge_override("12")          # the sibling with the box off
got = wrapped(ref_image_size="max")
check("a sibling's disarm does not cancel this node's arm", got == 3072,
      f"saw {got}")

rf.disarm_short_edge_override()
rf.arm_short_edge_override(3072, node_id="11")
rf.disarm_short_edge_override("11")          # this node's own disarm
got = wrapped(ref_image_size="max")
check("a node's own disarm does clear it", got == BASE, f"saw {got}")

print("\narm_does_not_survive_an_unrelated_call:")
# The wrapper used to clear only on the armed path, so an arm that was never
# consumed sat in the module until some later prompt picked it up. It now
# clears on every call.
rf.disarm_short_edge_override()
rf.arm_short_edge_override(3072, node_id="11")
first = wrapped(ref_image_size="max")
second = wrapped(ref_image_size="max")
third = wrapped(ref_image_size="max")
check("armed once, applied once",
      first == 3072 and second == BASE and third == BASE,
      f"saw {first}, {second}, {third}")

print("\ndownstream_ref_image_size:")
# The 'match' detector that makes the node's own log honest. A false alarm on
# every render would be worse than the silence it replaces, so "cannot tell"
# must read as None rather than as a warning.
PROMPT = {
    "9": {"class_type": "MiniMaxH3ReferenceToVideo",
          "inputs": {"ref_images.ref_image_0": ["13", 0], "ref_image_size": "match"}},
    "13": {"class_type": "MiniMaxH3ReferenceFit", "inputs": {}},
}
check("reads the consumer's setting", rf._downstream_ref_image_size(PROMPT, "13") == "match")
check("None when this node feeds nothing", rf._downstream_ref_image_size(PROMPT, "99") is None)
check("None without a prompt", rf._downstream_ref_image_size(None, "13") is None)
MAXP = {"9": {"class_type": "MiniMaxH3ReferenceToVideo",
              "inputs": {"ref_images.ref_image_0": ["13", 0], "ref_image_size": "max"}}}
check("reads 'max' when set", rf._downstream_ref_image_size(MAXP, "13") == "max")
DEFP = {"9": {"class_type": "MiniMaxH3ReferenceToVideo",
              "inputs": {"ref_images.ref_image_0": ["13", 0]}}}
check("absent key reads as core's 'match' default",
      rf._downstream_ref_image_size(DEFP, "13") == "match")

print("\ninstall_is_idempotent:")
original = core.MiniMaxH3ReferenceToVideo.__dict__.get("execute")
try:
    rf._install_wrapper()
    once = core.MiniMaxH3ReferenceToVideo.__dict__.get("execute")
    rf._install_wrapper()
    twice = core.MiniMaxH3ReferenceToVideo.__dict__.get("execute")
    check("second install is a no-op", once is twice)
    inner = twice.__func__ if isinstance(twice, classmethod) else twice
    check("the wrapper carries its marker",
          getattr(inner, rf._WRAP_MARKER, False))
finally:
    if original is not None:
        core.MiniMaxH3ReferenceToVideo.execute = original

if failures:
    print(f"\n{len(failures)} failed: {', '.join(failures)}")
    raise SystemExit(1)
print("\nthe override is scoped to one call")
