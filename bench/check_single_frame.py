#!/usr/bin/env python3
"""Check the single-frame shim changes `length=1` and provably nothing else.

`single_frame.apply()` lowers core's H3 `length` floor from 5 to 1 so H3 can be
driven as a single-image edit model. The risk it carries is not that it fails --
a failed patch is visible the moment a render asks for one frame and gets five.
The risk is that it succeeds and *also* moves something else, because the clamp
it relaxes (`max(5, length)` -> `max(1, length)`) sits underneath every length
in the node, and 2, 3 and 4 were never reachable as themselves before.

So the headline case is not "1 works". It is "1 is the ONLY input whose answer
moved", asserted by walking the node's entire declared domain and counting the
differences. One difference passes. Two fail, whichever they are.

Claims, i.e. what breaks if a case is deleted:
  exactly one answer moves   pristine vs patched over the whole legal domain
                             (1..node max) differ at exactly one input, and it
                             is 1. This is the whole file
  1 becomes a single frame   patched `temporal_shape(1)` is (1 frame, 1 latent
                             step, 2 audio steps); stock is (5, 2, 8)
  2..4 still snap to 5       called out separately from the sweep because it is
                             the specific region the relaxed clamp exposes, and
                             a sweep that regressed here would still be a sweep
  the guard can refuse       CONTROL: hand `apply()` a deliberately wrong
                             implementation and it must refuse and leave the
                             module untouched. Without this the equivalence
                             sweep above is decoration -- it has never been
                             shown failing
  refusal leaves nothing     after a refusal the module still answers exactly
                             as it did before, with no partial patch
  the floor reaches the API  all three nodes report min=1 through INPUT_TYPES,
                             which is the surface ComfyUI validates a submitted
                             prompt against. A schema patched any shallower
                             looks applied and still rejects length=1
  retirement works           against a module that already supports single
                             frames, `apply()` does nothing and says so
  a broken module is safe    a module missing the functions gets no patch, no
                             exception, and an unhealthy report
  applying twice is a no-op  the second call cannot double-wrap anything
  nothing else calls in      core has exactly one call site for these three
                             functions. The containment argument -- "the only
                             way to reach the T=1 branch is a length widget" --
                             is true today and this is what notices when it
                             stops being
  every copy is found        ComfyUI loads `comfy_extras` files under a
                             file-path module name and the directory has no
                             __init__.py, so a dotted import builds a SECOND
                             module object. Patching one and not the other
                             looks exactly like success
  LIVE: the server agrees    the running ComfyUI reports min=1 through
                             /object_info. Skipped, loudly, with no server.
                             This is the only case here that reads the surface
                             ComfyUI actually validates against, and on
                             2026-08-15 it was the only thing that could tell
                             a patched module from a patched *copy* of one

**This one TOUCHES CUDA, unlike most of `bench/check_*.py`.** It has to import
`comfy_extras.nodes_minimax_h3` to patch it, that module opens with
`import nodes`, and `nodes` pulls in `comfy.model_management`, which
initialises the device at import (`model_management.py`, `get_total_memory` at
module scope). With a render resident that raises
`torch.AcceleratorError: CUDA error: out of memory` before the first case runs
-- which reads as a regression and is not one. Free the GPU first:

    curl -X POST localhost:8188/free -H 'Content-Type: application/json' \\
         -d '{"unload_models": true, "free_memory": true}'
    PYTHONPATH=/path/to/ComfyUI python bench/check_single_frame.py

No model weights are loaded and the LIVE case is the only one wanting a server.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parents[1]
# ComfyUI root must win: `comfy_extras/nodes_minimax_h3` opens with a bare
# `import nodes` and this repo has its own nodes.py. Same ordering rule as
# check_schema_defaults.py, and the same three debugging rounds behind it.
sys.path.insert(0, str(REPO.parent))
sys.path.insert(0, str(COMFY))


def _load_shim():
    """single_frame.py by path, NOT through the package.

    Importing the package runs `__init__.py`, which applies the shim -- and
    this check has to hold the unpatched module in its hands to compare
    against. Loading the module standalone is what keeps the ordering ours.
    """
    spec = importlib.util.spec_from_file_location(
        "_h3_single_frame_under_test", REPO / "single_frame.py")
    assert spec and spec.loader, "single_frame.py did not load"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pristine_from_source(path):
    """An INDEPENDENT baseline: the three grid functions re-executed from source.

    Holding references to the module's own function objects does not work, and
    the way it fails is subtle enough to have shipped. `temporal_shape`'s body
    resolves `align_frame_count` and `video_latent_t` through module globals --
    the very names `apply()` replaces -- so a saved reference to the original
    `temporal_shape` still calls the PATCHED helpers. Both sides of the
    comparison then move together and the sweep can only see changes made
    inside `temporal_shape` itself. Reproduced 2026-08-15: with
    `_make_align_frame_count` mutated to return 999 at n == 200, the headline
    case still reported `moved == [1]`.

    Re-executing from source gives functions whose globals are a namespace
    nothing patches. Extracted with `ast` rather than imported, so the baseline
    also costs no torch, no CUDA and no ComfyUI import.
    """
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
                "align_frame_count", "video_latent_t", "temporal_shape"):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") in ("FPS", "AUDIO_LATENT_FPS")
                for t in node.targets):
            keep.append(node)
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    missing = [n for n in ("align_frame_count", "video_latent_t",
                           "temporal_shape", "FPS") if n not in ns]
    assert not missing, (
        f"could not rebuild a baseline from {path}: missing {missing}. "
        f"Without one this file cannot compare against anything.")
    return ns


def _snapshot(mod):
    return (mod.align_frame_count, mod.video_latent_t, mod.temporal_shape)


def _restore(mod, snap):
    mod.align_frame_count, mod.video_latent_t, mod.temporal_shape = snap
    if hasattr(mod, "_h3_explorations_single_frame"):
        delattr(mod, "_h3_explorations_single_frame")


def main():
    failures = []

    def check(name, fn):
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {exc}")

    print("single-frame shim: length=1 enabled, everything else identical")

    shim = _load_shim()
    import comfy_extras.nodes_minimax_h3 as core

    assert not getattr(core, "_h3_explorations_single_frame", False), (
        "the core module is already patched before this check ran; it cannot "
        "compare against a pristine baseline and would pass trivially")

    # The baseline: core's grid functions re-executed from source into their
    # own namespace, so nothing `apply()` does can reach them. Comparisons are
    # against these rather than against numbers typed here -- a check that
    # asserts its own arithmetic agrees with itself, not with core.
    base = _pristine_from_source(pathlib.Path(core.__file__))
    pristine_align = base["align_frame_count"]
    pristine_latent_t = base["video_latent_t"]
    pristine_shape = base["temporal_shape"]
    domain = shim.unchanged_domain(core)
    stock_one = pristine_shape(1)

    report = shim.apply(module=core, log=False)
    assert report.applied and report.healthy, f"shim did not apply: {report.line()}"
    print(f"        ({report.line()})")
    print(f"        (domain swept: length {min(domain)}..{max(domain)}, "
          f"{len(domain)} values, 1 excluded as the one meant to change)")

    def exactly_one_answer_moves():
        # The swept domain plus the one value it excludes, so this counts
        # changes over EVERY reachable length rather than over the ones the
        # shim chose to look at.
        moved = [n for n in sorted(set(domain) | {1})
                 if pristine_shape(n) != core.temporal_shape(n)]
        assert moved == [1], (
            f"expected exactly one length to change and it to be 1; "
            f"{len(moved)} changed: {moved[:12]}"
            f"{'...' if len(moved) > 12 else ''}")

    def one_is_a_single_frame():
        frames, latent_t, audio_t = core.temporal_shape(1)
        assert (frames, latent_t) == (1, 1), (
            f"length=1 gives {frames} frames / {latent_t} latent steps, "
            f"not a single frame")
        assert audio_t == round(1 / core.FPS * core.AUDIO_LATENT_FPS), (
            f"audio steps {audio_t} is not what one frame's duration asks for")
        assert stock_one == (5, 2, 8), (
            f"stock length=1 was {stock_one}, not the (5, 2, 8) this check "
            f"was written against -- core's arithmetic moved, re-derive")

    def two_to_four_still_snap_to_five():
        # The region the relaxed clamp exposes: unreachable as itself before
        # (core clamped to 5), reachable now, and it must still round up.
        for n in (2, 3, 4):
            assert core.temporal_shape(n) == (5, 2, 8), (
                f"length={n} gives {core.temporal_shape(n)}; it must still "
                f"snap to the 5-frame grid step, as it did before the patch")
            assert core.align_frame_count(n) == 5
            assert core.video_latent_t(n) == 2

    def the_guard_can_refuse():
        # CONTROL. A patch that is wrong in the way a hand-written one would be
        # wrong: it lifts the clamp AND collapses everything below the old
        # floor to one frame. The equivalence sweep inside apply() is the only
        # thing standing between that and a silent behaviour change at 2..4.
        saved = _snapshot(core)
        _restore(core, (pristine_align, pristine_latent_t, pristine_shape))
        original_maker = shim._make_temporal_shape
        try:
            shim._make_temporal_shape = lambda mod: (
                lambda length: (1, 1, 2) if length <= 5
                else original_maker(mod)(length))
            bad = shim.apply(module=core, log=False)
            assert not bad.applied, "apply() installed a patch that changes 2..5"
            assert not bad.healthy, "a refusal must not report as healthy"
            assert "refusing to patch" in bad.reason, (
                f"refused for the wrong reason: {bad.reason}")
            assert core.temporal_shape is pristine_shape, (
                "refusal left the module's temporal_shape replaced")
            assert core.temporal_shape(2) == (5, 2, 8), (
                "refusal left core answering differently at length=2")
            assert not getattr(core, "_h3_explorations_single_frame", False), (
                "refusal still marked the module as patched")
        finally:
            shim._make_temporal_shape = original_maker
            _restore(core, saved)
            core._h3_explorations_single_frame = True

    def the_floor_reaches_the_api():
        floors = {name: shim.schema_length_min(getattr(core, name))
                  for name in shim.NODE_CLASSES}
        assert set(floors) == set(shim.NODE_CLASSES), floors
        bad = {n: f for n, f in floors.items() if f != 1}
        assert not bad, (
            f"INPUT_TYPES still reports a floor above 1, so ComfyUI will "
            f"reject length=1 on: {bad}")

    def retirement_works():
        class _Already:
            FPS, AUDIO_LATENT_FPS = core.FPS, core.AUDIO_LATENT_FPS
            align_frame_count = staticmethod(lambda n: max(1, core.align_frame_count(n)) if n > 1 else 1)
            video_latent_t = staticmethod(lambda n: 1 if n <= 1 else core.video_latent_t(n))
            temporal_shape = staticmethod(lambda n: (1, 1, 2) if n <= 1 else core.temporal_shape(n))

            class EmptyMiniMaxH3LatentAV:
                @staticmethod
                def INPUT_TYPES():
                    return {"required": {"length": ("INT", {"min": 1, "max": 3600})}}

        out = shim.apply(module=_Already, log=False)
        assert not out.applied and out.healthy, out
        assert "already supports" in out.reason, out.reason

    def a_broken_module_is_safe():
        class _Wrong:
            class MiniMaxH3ImageToVideo:
                @staticmethod
                def INPUT_TYPES():
                    return {"required": {"length": ("INT", {"min": 5, "max": 3600})}}

        out = shim.apply(module=_Wrong, log=False)
        assert not out.applied, out
        assert not out.healthy, "an unrecognised module must report unhealthy"
        assert "not the shape this shim knows" in out.reason, out.reason

    def applying_twice_is_a_noop():
        before = core.temporal_shape
        again = shim.apply(module=core, log=False)
        assert again.applied and "already applied" in again.reason, again.reason
        assert core.temporal_shape is before, "second apply() rewrapped the module"
        tips = [t for t in (getattr(i, "tooltip", "") or ""
                            for i in core.EmptyMiniMaxH3LatentAV.define_schema().inputs)
                if shim._TOOLTIP_NOTE.strip() in t]
        assert len(tips) == 1 and tips[0].count("lowered this floor") == 1, (
            "the tooltip note was applied more than once")

    def nothing_else_calls_in():
        names = ("align_frame_count", "video_latent_t", "temporal_shape")
        pattern = re.compile("|".join(names))
        hits = {}
        for path in list((COMFY / "comfy").rglob("*.py")) + \
                list((COMFY / "comfy_extras").rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if pattern.search(text):
                hits[path.relative_to(COMFY).as_posix()] = sum(
                    len(re.findall(n, text)) for n in names)
        assert list(hits) == ["comfy_extras/nodes_minimax_h3.py"], (
            f"these functions are referenced outside the H3 node module, so "
            f"'the only way in is a length widget' no longer holds: {hits}")

    def every_copy_is_found():
        # A stand-in for the dotted second copy, without paying for a real
        # re-import: same file on disk, different module object and name. If
        # discovery misses it, the running server keeps executing an unpatched
        # temporal_shape while the log reports success.
        import types

        stub = types.ModuleType("_h3_second_copy_probe")
        stub.__file__ = core.__file__
        stub.FPS, stub.AUDIO_LATENT_FPS = core.FPS, core.AUDIO_LATENT_FPS
        stub.align_frame_count = pristine_align
        stub.video_latent_t = pristine_latent_t
        stub.temporal_shape = pristine_shape
        sys.modules[stub.__name__] = stub
        try:
            found = shim.module_copies()
            assert any(m is stub for m in found), (
                f"discovery missed a second copy of {core.__file__}; it found "
                f"{[getattr(m, '__name__', '?') for m in found]}")
            assert any(m is core for m in found), "discovery missed the real module"
            out = shim.apply(log=False)
            assert out.applied and out.healthy, out.line()
            assert stub.temporal_shape(1)[0] == 1, (
                "the second copy was found but left unpatched, which is the "
                "shape of the 2026-08-15 bug")
        finally:
            sys.modules.pop(stub.__name__, None)

    def resolution_refuses_without_support():
        """A length of 1 on a ComfyUI that cannot render one frame must RAISE.

        ComfyUI validates a widget's `min` only on literal values. The shipped
        image graph wires `length` over a link from this node, so core never
        checks it -- measured 2026-08-15 with the shim disabled, the graph was
        accepted and rendered FIVE frames through the single-image VAE with
        nothing said. `MiniMaxH3Resolution` is what converts that into a
        refused render, and this is the case that keeps it doing so.
        """
        import importlib.util as _ilu

        # resolution.py falls back to a bare `import h3_rules` when it is not
        # loaded as a package member, so the repo root has to be importable.
        # APPENDED, never inserted: this directory also contains a `nodes.py`,
        # and ComfyUI's must keep winning that name.
        if str(REPO) not in sys.path:
            sys.path.append(str(REPO))
        spec = _ilu.spec_from_file_location("_h3_resolution_probe", REPO / "resolution.py")
        assert spec and spec.loader
        res = _ilu.module_from_spec(spec)
        spec.loader.exec_module(res)

        shape = {"shape": "tall", "tall_resolution": "768x1152  2/3  864 tok/frame  0.73x"}
        # Control: with single-frame support present, length=1 must go through.
        res.MiniMaxH3Resolution.execute(shape, length=1)

        # Now simulate a ComfyUI without it, by restoring the pristine function
        # on the module the probe consults.
        saved = core.temporal_shape
        core.temporal_shape = pristine_shape
        try:
            try:
                res.MiniMaxH3Resolution.execute(shape, length=1)
            except RuntimeError as exc:
                assert "single frame" in str(exc), f"refused for the wrong reason: {exc}"
            else:
                raise AssertionError(
                    "length=1 was accepted on a ComfyUI that clamps it to 5; "
                    "the graph would render 5 frames through a 1-frame VAE")
            # and a normal video length must still be unaffected by the guard
            res.MiniMaxH3Resolution.execute(shape, length=124)
        finally:
            core.temporal_shape = saved

    def live_server_agrees():
        import json
        import urllib.request

        base = "http://127.0.0.1:8188"
        try:
            urllib.request.urlopen(f"{base}/system_stats", timeout=2).read()
        except Exception as exc:
            raise AssertionError(
                f"SKIPPED -- no ComfyUI at {base} ({type(exc).__name__}). This "
                f"is the only case that reads what ComfyUI validates against; "
                f"treat the rest of this file as unverified against a server") \
                from None
        bad = {}
        for name in shim.NODE_CLASSES:
            with urllib.request.urlopen(f"{base}/object_info/{name}", timeout=5) as fh:
                info = json.load(fh)
            spec = info[name]["input"]["required"]["length"][1]
            if spec.get("min") != 1:
                bad[name] = spec.get("min")
        assert not bad, (
            f"the running server still reports a floor above 1: {bad}. The "
            f"shim patched something, but not what ComfyUI serves")

    check("exactly one length's answer moves, and it is 1", exactly_one_answer_moves)
    check("length=1 is one frame, one latent step", one_is_a_single_frame)
    check("length 2..4 still snap to 5", two_to_four_still_snap_to_five)
    check("CONTROL: the equivalence guard refuses a wrong patch", the_guard_can_refuse)
    check("the floor reaches INPUT_TYPES on all three nodes", the_floor_reaches_the_api)
    check("retirement: a module that already supports 1 is left alone", retirement_works)
    check("an unrecognised module is refused, not crashed", a_broken_module_is_safe)
    check("applying twice is a no-op", applying_twice_is_a_noop)
    check("nothing outside the H3 node module calls these", nothing_else_calls_in)
    check("every loaded copy of the module is found and patched", every_copy_is_found)
    check("the Resolution node refuses length=1 on an unpatched ComfyUI",
          resolution_refuses_without_support)
    check("LIVE: the running server reports a floor of 1", live_server_agrees)

    print(f"\n{len(failures)} failure(s)" if failures else
          "\nall ok -- length=1 renders one frame, every other length is untouched")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
