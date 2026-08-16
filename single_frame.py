"""TEMPORARY. Lifts ComfyUI's 5-frame floor on the stock H3 nodes, in memory.

    ############################################################
    #  THIS FILE PATCHES COMFYUI CORE AND IS MEANT TO BE DELETED.
    #  It exists only until ComfyUI ships the same change.
    #    tracking:  Comfy-Org/ComfyUI#15644
    #    ends when: `/object_info` reports length min=1 without us,
    #               at which point `apply()` retires itself and the
    #               startup log says to delete this file.
    #    disable:   H3_EXPLORATIONS_NO_SINGLE_FRAME=1
    #  Nothing else in this repo may grow a dependency on it: it is a
    #  patch to somebody else's module, not an interface of ours.
    ############################################################

**Read this before assuming it applies to anything else.** The only thing this
buys is `length=1`: one frame, one latent temporal step, decoded by the
single-image H3 VAE (`Mamad8/MiniMax-H3-Image-VAE`). It exists because H3 turns
out to be a strong single-image edit model when you generate exactly one frame,
and because ComfyUI's H3 nodes are the only video family in `comfy_extras` that
refuse to. Wan does not (16 length inputs, all `min=1`), nor Hunyuan (3), nor
Cosmos (3); LTX splits 1 and 9, Mochi floors at 7. H3 at 5 is the outlier.

Upstream tracking: Comfy-Org/ComfyUI#15644, open, no PR at the time of writing.

**What this does NOT do, and please do not read it as doing.** It does not make
H3 an image model in general, it does not touch anything at `length >= 2`, and
it is not a quality claim. At `length=1`:

  - the stock video VAE is the wrong decoder. It was never asked for a T=1
    latent before this and the community reports grid artifacts; the image VAE
    above is what makes the path worth having. That VAE's own README says it
    materially regresses multi-frame reconstruction, so it must never be wired
    into a video graph.
  - the audio stream is still there and is 2 latent frames of nothing
    (`round(1/24 * 40)`). Decode it and you get ~0.04s. Every shipped
    single-frame graph leaves the audio decoder out rather than wiring noise.
  - a `last_frame` keyframe has nowhere to land: `MiniMaxH3ImageToVideo` pins it
    at `frame_count - 1`, which is frame 0 in a 1-frame video, i.e. on top of
    `first_frame`. Reference images have no such problem, which is why the
    single-image path here is ref2v and not fl2v.

**One thing this DOES widen, and it is not the arithmetic.** Lowering the
schema floor means prompt validation now accepts `length` 2, 3 and 4, which it
rejected outright before. They render exactly as 5 does, because that is what
they always snapped to -- but "nothing else changes" is a claim about output,
not about what the API admits. Say it that way.

**Why every other length is safe is verified at apply time, not argued.** The two
grid functions keep upstream's own body for every count above 1 -- the patch
delegates rather than reimplementing, so an upstream change to the 17k+5
arithmetic still flows through. `temporal_shape` is the one function that has
to be rewritten, because the clamp is inside it, and `apply()` refuses to
install anything unless the rewrite reproduces the original exactly across
`unchanged_domain()` -- the node's ENTIRE declared range, walked at every
startup, not a sample of it. If it disagrees at any length, nothing is patched
and the reason is logged. That is the check this module trusts; the sentence
you are reading is not.

**Retirement.** When upstream lands the same change, `apply()` sees a floor of
1 already in place, does nothing, and says so in the log. Nothing here needs to
be deleted for that to be correct -- delete it when the log line has been saying
"already supports single frames" for a while.

**Escape hatch.** Set `H3_EXPLORATIONS_NO_SINGLE_FRAME=1` in the environment and
this module does nothing at all, leaving core exactly as shipped.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import NamedTuple

logger = logging.getLogger(__name__)

MODULE = "comfy_extras.nodes_minimax_h3"
DISABLE_ENV = "H3_EXPLORATIONS_NO_SINGLE_FRAME"

# The three stock nodes that take `length`. Order is display order, not
# significance; a missing one is skipped rather than fatal.
NODE_CLASSES = ("EmptyMiniMaxH3LatentAV", "MiniMaxH3ImageToVideo",
                "MiniMaxH3ReferenceToVideo")

_MARK = "_h3_explorations_single_frame"

# The patch must leave EXACTLY one input's answer different: length=1. That is
# checked by walking the node's whole legal domain rather than a sample of it,
# because the interesting region is 2, 3 and 4 -- the counts that were never
# reachable as themselves (core clamps them to 5) and that this patch moves the
# clamp underneath. A sample that happened to skip them would prove nothing
# about the only place a silent change could hide. The full sweep is ~30k loop
# steps and runs in single-digit milliseconds at import.
#
# The upper bound is read off the node's own schema so it tracks upstream; this
# is only the fallback for a schema that does not declare one.
_DOMAIN_MAX_FALLBACK = 3600

# The sweep starts below zero on purpose. `length` arrives over links as well
# as widgets and links are never min-validated, so the reachable domain is not
# the widget's range -- `MiniMaxH3KeyframeCanvas` emits 0 as its "opt out"
# sentinel, straight into the stock node.
_DOMAIN_MIN = -8

_TOOLTIP_NOTE = (
    " -- ComfyUI-h3-explorations lowered this floor from 5 to 1 for the "
    "single-image edit path. length=1 renders ONE frame and needs the "
    "single-image H3 VAE to decode it; the stock video VAE is the wrong "
    "decoder there. Nothing above 1 is changed."
)


class Report(NamedTuple):
    """What `apply()` did, in a form a check can assert on.

    `applied` is whether `length=1` works now. `healthy` is whether the outcome
    is one we expect -- both a successful patch and a retired shim are healthy,
    while a refusal or a half-applied patch is not. They are separate fields
    because "did nothing" is the correct answer in two situations and a
    reportable one in three, and a single boolean cannot carry that.
    """

    applied: bool
    reason: str
    functions: tuple[str, ...] = ()
    schemas: tuple[str, ...] = ()
    healthy: bool = True

    def line(self):
        detail = ""
        if self.functions or self.schemas:
            detail = (f" ({len(self.functions)} functions, "
                      f"{len(self.schemas)} node schemas)")
        return f"{self.reason}{detail}"


def _registered_classes():
    """The H3 node classes ComfyUI actually serves, from its own registry.

    `nodes` is read out of `sys.modules` rather than imported: this repo has
    its own `nodes.py`, and importing the name from here could bind ours for
    the rest of the process. Under ComfyUI the real one is always loaded.
    """
    mapping = getattr(sys.modules.get("nodes"), "NODE_CLASS_MAPPINGS", None) or {}
    return {name: mapping[name] for name in NODE_CLASSES if name in mapping}


def module_copies(module=None):
    """EVERY loaded copy of core's H3 node module, newest registry copy first.

    There is normally more than one, and patching the wrong one looks exactly
    like success. `load_custom_node` registers each `comfy_extras` file under a
    **file-path** module name (`nodes.py`: `sys_module_name =
    os.path.splitext(module_path)[0]`), and `comfy_extras/` ships no
    `__init__.py`, so it is a PEP 420 namespace package. A dotted
    `import comfy_extras.nodes_minimax_h3` -- which this repo's own
    `resolution.py` and `preflight.py` do, and so does anything else reaching
    for `adapt_canvas` -- therefore builds a SECOND, independent module object
    with its own copy of the grid functions.

    Patching only the dotted copy leaves ComfyUI serving and executing the
    other one: `/object_info` keeps reporting a floor of 5 while the shim
    cheerfully logs success. That happened on 2026-08-15 and was invisible to
    an in-process check, because the check handed the module in itself and so
    agreed with the bug.

    So targets are resolved by identity, not by name: start from the classes in
    `NODE_CLASS_MAPPINGS` (the copy that certainly matters), take the file they
    came from, and collect every module in `sys.modules` loaded from that same
    file.
    """
    if module is not None:
        return [module]

    found = {}
    for cls in _registered_classes().values():
        mod = sys.modules.get(getattr(cls, "__module__", "") or "")
        if mod is not None:
            found[id(mod)] = mod
    dotted = sys.modules.get(MODULE)
    if dotted is not None:
        found.setdefault(id(dotted), dotted)

    files = {os.path.realpath(f) for f in
             (getattr(m, "__file__", None) for m in found.values()) if f}
    if files:
        for mod in list(sys.modules.values()):
            path = getattr(mod, "__file__", None)
            if path and os.path.realpath(path) in files:
                found.setdefault(id(mod), mod)
    return list(found.values())


def _make_align_frame_count(original):
    def align_frame_count(n):
        # One frame is one frame. Everything else is upstream's, unchanged --
        # including whatever upstream does to it later.
        if n <= 1:
            return 1
        return original(n)

    return align_frame_count


def _make_video_latent_t(original):
    def video_latent_t(frame_count):
        # A single pixel frame is a single latent step: FRAME_PER_TOKEN[0] == 1
        # in comfy/ldm/minimax/model.py, so token 0 covers exactly frame 0.
        if frame_count <= 1:
            return 1
        return original(frame_count)

    return video_latent_t


def _make_temporal_shape(mod):
    def temporal_shape(length):
        # Reimplemented rather than delegated, because the `max(5, length)`
        # clamp this exists to relax lives inside it. Reads its constants and
        # its two helpers off the module so it composes with the patched pair
        # above and does not become a second copy of FPS / AUDIO_LATENT_FPS.
        #
        # **Only an explicit 1 is a single frame. 0 and below keep the old
        # 5-frame floor**, which `max(1, length)` -- the obvious spelling, and
        # the one the community patch uses -- would silently change. That is
        # not hypothetical here: `MiniMaxH3KeyframeCanvas` documents `length=0`
        # as "opt out", emits it as a LINK into the stock node, and link values
        # are never min-validated. Under `max(1, length)` that sentinel would
        # start rendering one frame instead of five.
        frame_count = mod.align_frame_count(length if length >= 1 else 5)
        duration = frame_count / mod.FPS
        return (frame_count, mod.video_latent_t(frame_count),
                round(duration * mod.AUDIO_LATENT_FPS))

    return temporal_shape


def supports_single_frame(mod) -> bool:
    """True when the module already answers `length=1` with one frame.

    The retirement condition. Asked of behaviour rather than of a version
    number, so it is equally true of an upstream fix, of this shim, and of a
    hand-patched file.
    """
    try:
        return (mod.align_frame_count(1) == 1
                and mod.video_latent_t(1) == 1
                and mod.temporal_shape(1)[0] == 1)
    except Exception:
        return False


def _length_input(schema):
    for inp in getattr(schema, "inputs", ()) or ():
        if getattr(inp, "id", None) == "length":
            return inp
    return None


def _schema_length_spec(node_cls, key):
    """One declared property of this node's `length` input, or None.

    Read through `INPUT_TYPES()` rather than off the class, because that is the
    surface ComfyUI validates a submitted prompt against and serves from
    `/object_info`. A schema patched anywhere shallower than this would look
    applied and still reject `length=1`.
    """
    try:
        spec = node_cls.INPUT_TYPES()
    except Exception:
        return None
    for section in ("required", "optional"):
        entry = (spec.get(section) or {}).get("length")
        if isinstance(entry, tuple) and len(entry) > 1 and isinstance(entry[1], dict):
            return entry[1].get(key)
    return None


def schema_length_min(node_cls):
    return _schema_length_spec(node_cls, "min")


def schema_length_max(node_cls):
    return _schema_length_spec(node_cls, "max")


def unchanged_domain(mod, node_classes=NODE_CLASSES):
    """Every length whose answer this patch must not move. 1 is the exception.

    Runs from `_DOMAIN_MIN` to the node's own declared maximum, so it covers
    what a user or an API submission can get past validation -- and then some.
    It deliberately starts BELOW zero rather than at 2: `length` reaches these
    functions over links as well as widgets, links are never min-validated, and
    this repo's own keyframe node emits 0 as an "opt out" sentinel. A sweep
    starting at 2 would have proved nothing about the value most likely to
    arrive from our own graphs.
    """
    ceilings = [schema_length_max(getattr(mod, name))
                for name in node_classes if hasattr(mod, name)]
    ceilings = [c for c in ceilings if isinstance(c, int)]
    top = max(ceilings) if ceilings else _DOMAIN_MAX_FALLBACK
    return [n for n in range(_DOMAIN_MIN, top + 1) if n != 1]


_SCHEMA_MARK = "_h3_explorations_length_floor"


def _relax_schema(node_cls):
    """Lower this node's `length` floor to 1, at the schema it builds.

    Idempotent per class, read from `__dict__` rather than by attribute so an
    inherited mark can never make a real node look already done. Two different
    copies of the same class are two different objects and both get patched;
    the same object twice does not, which is what keeps the tooltip from
    growing a second copy of its own note on every call.
    """
    if node_cls.__dict__.get(_SCHEMA_MARK):
        return False
    original = node_cls.__dict__.get("define_schema")
    if original is None:
        # Inherited rather than declared: not a node we recognise, and
        # patching the base class would reach nodes we never meant to touch.
        return False
    fn = original.__func__ if isinstance(original, classmethod) else original

    def define_schema(cls, _fn=fn):
        schema = _fn(cls)
        inp = _length_input(schema)
        if inp is not None:
            inp.min = 1
            # step 1, and this is OUR choice rather than a precedent. Read
            # 2026-08-15: no `comfy_extras` family pairs `min=1` with `step=1`
            # -- Wan is min=1/step=4 at all 16 of its length inputs, Hunyuan
            # min=1/step=4, Cosmos min=1/step=4 or 8. The community patch uses
            # step=1 and this follows it, because H3's grid is 17n+5: core's
            # current 5/17 pairing walks 5, 22, 39, all on the grid, while
            # 1/17 would walk 1, 18, 35, none of which are. Rather than pick a
            # step that lands off-grid, let every value through and let
            # `align_frame_count` snap it, which it does regardless.
            # Nothing validates step server-side; this is the widget's
            # increment only.
            inp.step = 1
            inp.tooltip = (inp.tooltip or "") + _TOOLTIP_NOTE
        return schema

    node_cls.define_schema = classmethod(define_schema)
    setattr(node_cls, _SCHEMA_MARK, True)
    return True


def _prepare_one(mod):
    """Verify one module copy and return the installer for it, or a reason.

    Split from installation so `apply()` can check EVERY copy before touching
    any of them. Patching as it goes leaves the process half-patched when a
    later copy refuses -- reporting `applied=False` while one copy is already
    live, which is precisely the state this module claims never to install.
    """
    if getattr(mod, _MARK, False):
        return (lambda: None), "already applied", ()

    missing = [n for n in ("align_frame_count", "video_latent_t",
                           "temporal_shape", "FPS", "AUDIO_LATENT_FPS")
               if not hasattr(mod, n)]
    if missing:
        return None, (f"{MODULE} is not the shape this shim knows "
                      f"(missing {', '.join(missing)}); not patching"), ()

    align = _make_align_frame_count(mod.align_frame_count)
    latent_t = _make_video_latent_t(mod.video_latent_t)

    # Build the replacement against a stand-in that already carries the patched
    # pair, so the equivalence sweep below tests the trio as it will actually
    # run rather than a partly-patched module.
    class _Staged:
        FPS = mod.FPS
        AUDIO_LATENT_FPS = mod.AUDIO_LATENT_FPS
        align_frame_count = staticmethod(align)
        video_latent_t = staticmethod(latent_t)

    staged = _make_temporal_shape(_Staged)

    for n in unchanged_domain(mod):
        before, after = mod.temporal_shape(n), staged(n)
        if before != after:
            return None, (f"refusing to patch: length={n} would change from "
                          f"{before} to {after}"), ()
    if staged(1) != (1, 1, round(1 / mod.FPS * mod.AUDIO_LATENT_FPS)):
        return None, (f"refusing to patch: length=1 gives {staged(1)}, "
                      f"which is not a single frame"), ()

    def install():
        # setattr rather than attribute assignment: these are module globals
        # being replaced, which is the whole mechanism, and a type checker is
        # right that a module has no such declared attribute.
        setattr(mod, "align_frame_count", align)
        setattr(mod, "video_latent_t", latent_t)
        # Built against the MODULE, not against the `_Staged` stand-in used for
        # verification above. The stand-in freezes FPS and both helpers as class
        # attributes, so installing it would make the comment in
        # `_make_temporal_shape` false and would silently ignore anyone who
        # replaces `align_frame_count` later -- where core's own composes.
        # The two are equivalent at this point precisely because the helpers
        # have just been installed on the module.
        setattr(mod, "temporal_shape", _make_temporal_shape(mod))
        setattr(mod, _MARK, True)

    return install, "patched", ("align_frame_count", "video_latent_t",
                                "temporal_shape")


def apply(module=None, log=True) -> Report:
    """Patch every loaded copy of the module. Never raises; reports what it did.

    Called at pack import, where an exception would take every node in this
    repo down with it. A failure here has to cost the single-frame path and
    nothing else, so every exit is a Report.

    The schema half is applied to the classes ComfyUI has REGISTERED, not to
    whatever a module lookup returns -- see `module_copies` for why those are
    not always the same object, and for the day this distinction cost.
    """
    try:
        if os.environ.get(DISABLE_ENV):
            return _done(Report(False, f"disabled by {DISABLE_ENV}"), log)

        copies = module_copies(module)
        if not copies:
            return _done(Report(
                False, f"{MODULE} is not imported; nothing to patch",
                healthy=False), log)

        # Every class this shim can reach: the registry's (authoritative, and
        # what /object_info serves) plus each copy's own, deduplicated by
        # identity. A copy nobody registered still matters -- our own nodes
        # import from one.
        targets: dict[str, list] = {name: [] for name in NODE_CLASSES}
        for name, cls in _registered_classes().items():
            targets[name].append(cls)
        for mod in copies:
            for name in NODE_CLASSES:
                cls = getattr(mod, name, None)
                if cls is not None and not any(cls is seen for seen in targets[name]):
                    targets[name].append(cls)
        if not any(targets.values()):
            return _done(Report(False, "no H3 node classes found; not patching",
                                healthy=False), log)

        floors = [schema_length_min(cls) for group in targets.values() for cls in group]
        single_frame_works = (all(supports_single_frame(mod) for mod in copies)
                              and floors
                              and all(f is not None and f <= 1 for f in floors))
        # WHO made it work decides what to say, and the two answers are not
        # interchangeable: the retirement banner tells the reader to delete
        # this file. Saying that while we are the only reason length=1 works
        # would be an instruction to break the thing it is describing. So our
        # own mark is asked about FIRST, and upstream is credited only when no
        # copy carries it.
        ours = [mod for mod in copies if getattr(mod, _MARK, False)]
        if single_frame_works and ours:
            return _done(Report(True, "already applied", healthy=True), log)
        if single_frame_works:
            return _done(Report(
                False, "ComfyUI already supports single frames here; shim retired"), log)

        if os.environ.get("H3_SINGLE_FRAME_DEBUG"):
            logger.info("[h3] shim debug: copies=%s registry=%s targets=%s",
                        [getattr(m, "__name__", "?") for m in copies],
                        {k: id(v) for k, v in _registered_classes().items()},
                        {k: [id(c) for c in v] for k, v in targets.items()})

        # Two phases: verify EVERY copy, then install. A single refusal aborts
        # the whole thing with nothing touched, so "not patched" never means
        # "some copies patched" -- the half-applied state the docstring
        # promises never to leave behind.
        functions: tuple[str, ...] = ()
        installers = []
        for mod in copies:
            install, reason, patched = _prepare_one(mod)
            if install is None:
                return _done(Report(False, reason, healthy=False), log)
            installers.append(install)
            functions = functions or patched
        for install in installers:
            install()

        schemas = tuple(
            name for name, group in targets.items()
            if any([_relax_schema(cls) for cls in group]))

        # Confirm at the surface ComfyUI reads, not at the one we wrote to.
        # This is the assertion that would have caught the wrong-copy bug.
        still_floored = sorted({
            name for name, group in targets.items() for cls in group
            if (schema_length_min(cls) or 5) > 1})
        unpatched = [getattr(mod, "__name__", "?") for mod in copies
                     if mod.temporal_shape(1)[0] != 1]
        if still_floored or unpatched:
            return _done(Report(
                True, f"partly applied -- floor above 1 on "
                      f"{still_floored or 'none'}, functions unpatched in "
                      f"{unpatched or 'none'}", functions, schemas,
                healthy=False), log)

        return _done(Report(
            True, f"single-frame (length=1) enabled on ComfyUI's H3 nodes for "
                  f"the image-edit path; every length above 1 is unchanged "
                  f"[{len(copies)} module copies]",
            functions, schemas), log)
    except Exception as exc:  # pragma: no cover - defensive by intent
        return _done(Report(False, f"not patching: {type(exc).__name__}: {exc}",
                            healthy=False), log)


#: Printed in full whenever the shim actually patches something. It is a
#: monkey-patch of ComfyUI core, and the one failure mode nobody notices is a
#: temporary thing quietly becoming permanent -- so the console says out loud
#: what it is, what to watch for, and how it ends. Do not shorten this to one
#: line: the whole point is that it is hard to skim past.
_ACTIVE_BANNER = (
    "TEMPORARY PATCH TO COMFYUI CORE IS ACTIVE.\n"
    "    what   ComfyUI's stock MiniMax H3 nodes accept length=1 (single "
    "frame), which core normally floors at 5.\n"
    "    why    H3 is a strong single-image edit model at one frame. H3 is the "
    "only video family in comfy_extras with a floor above 1.\n"
    "    scope  Only length=1 RENDERS differently: every other input from -8 "
    "to the node maximum is verified identical at load, or nothing is patched.\n"
    "           The floor also makes prompt validation ACCEPT 2-4, which it "
    "rejected before; those snap to 5 exactly as 5 does.\n"
    "    needs  the single-image H3 VAE to decode (Mamad8/MiniMax-H3-Image-VAE) "
    "-- never wire that VAE into a video graph.\n"
    "    ENDS   when ComfyUI ships this upstream (Comfy-Org/ComfyUI#15644). "
    "This shim then retires itself automatically and logs that instead;\n"
    "           when you see that line, DELETE single_frame.py and its call in "
    "__init__.py. Set H3_EXPLORATIONS_NO_SINGLE_FRAME=1 to disable it now."
)

_RETIRED_BANNER = (
    "ComfyUI now supports single frames on its own -- this shim did nothing.\n"
    "    DELETE custom_nodes/ComfyUI-h3-explorations/single_frame.py and the "
    "single_frame.apply() call in that pack's __init__.py. Its job is done."
)


def _done(report: Report, log: bool) -> Report:
    if log:
        if report.applied and report.healthy and report.functions:
            logger.info("[h3] single-frame shim: %s\n    (%s)",
                        _ACTIVE_BANNER, report.line())
        elif "retired" in report.reason:
            logger.info("[h3] single-frame shim: %s", _RETIRED_BANNER)
        else:
            (logger.info if report.healthy else logger.warning)(
                "[h3] single-frame shim: %s", report.line())
    return report
