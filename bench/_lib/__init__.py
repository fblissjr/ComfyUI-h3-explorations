"""The things a `bench/` check kept writing for itself.

    from _lib import bootstrap, needs, case, refused, finish

A check runs as `python bench/check_x.py`, so `bench/` is `sys.path[0]` and
`_lib` imports with no setup. `docs/method/checks.md` "Running them" says what each
exit code means; this file is where two of them are produced.

`bootstrap()`
    Makes ComfyUI core importable and keeps it off a card it cannot see.
    Importing almost anything under `comfy` or `comfy_extras` pulls
    `comfy.model_management`, which picks a torch device AT IMPORT and raises
    "No CUDA GPUs are available" when `CUDA_VISIBLE_DEVICES=` hides the card.
    The switch is `comfy.cli_args.args.cpu = True`, set before that import.
    Most checks that import core set it by hand; the ones that forgot read as
    red in a masked sweep on 2026-10-05 although none of them uses a device
    (the list is in that day's CHANGELOG entry).
    `workflows/build_workflows.py::_core_cpu_when_no_card` is the same switch
    for the generator.

`needs(what, present)`
    Exit 2, "nothing graded", when something the check cannot run without is
    not there: a card, a dataset, an environment variable. A missing resource
    is not a failure of the thing under test, and a check that returns 0
    without grading reads as a pass: `check_correctness.py` and
    `check_clone_v_wiring.py` both printed "skipping" and returned 0.

`server_memory_mode()`
    For a script that calls core's nodes in process on the card: sets up the
    dynamic VRAM layer the server sets up at start. Its docstring says why
    that is not optional there.

`case(name, fn)`, `refused(target, *words)` and `finish()`
    One `ok` / `FAIL` line per case, one refusal assertion helper that rejects
    exit 0, bare exit codes, uncaught crashes and Python tracebacks, and one
    exit code: 1 when any case failed, 2 when none failed but one could not run
    (`skip(reason)` from inside it), 0 otherwise. For a new check, or an old
    one whose own loop was this already; nobody is asked to port a check that
    has a harness.

Nothing here imports torch or ComfyUI at module level, so a check that only
wants `needs` pays for nothing else.
"""

from __future__ import annotations

import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent
# The same directory `workflows/h3_config.py::COMFY_ROOT` names. Derived again
# rather than imported: h3_config is on no check's path until the check puts it
# there, and this module has to work before any path is set up.
COMFY = REPO.parents[1]

EXIT_OK, EXIT_FAIL, EXIT_NOT_GRADED = 0, 1, 2


# ------------------------------------------------------------------ bootstrap

def card_visible() -> bool:
    """Whether torch can see a CUDA device. False under `CUDA_VISIBLE_DEVICES=`."""
    import torch
    return bool(torch.cuda.is_available())


def bootstrap(*, cpu: bool | None = None, front: bool = True) -> Path:
    """Put the ComfyUI root on `sys.path` and choose core's device path.

    Call it BEFORE the first `import comfy...`; it raises if core's model
    management is already imported and the switch would arrive too late.

    cpu=True   core takes its CPU path whatever is visible. For a check that
               never touches a device, so it behaves the same masked or not
               and does not open a CUDA context beside somebody's render.
    cpu=None   CPU path only when no card is visible; with a card, core is
               left alone. The generator's rule.
    cpu=False  core is left alone. For a check that needs the card and has
               already said so with `needs(...)`.

    front=True puts the root first, which is what the checks that inserted it
    themselves did: this pack's `nodes.py` shares a name with ComfyUI's, and
    core has to win that lookup. front=False appends it only when it is absent
    and never reorders, for the two checks that used to take it from
    `PYTHONPATH` and whose import order with a card nobody has re-verified.

    Returns the ComfyUI root.
    """
    root = str(COMFY)
    if front:
        while root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    elif root not in sys.path:
        sys.path.append(root)

    if cpu is None:
        cpu = not card_visible()
    if cpu:
        import comfy.cli_args
        if "comfy.model_management" in sys.modules and not comfy.cli_args.args.cpu:
            raise RuntimeError(
                "bootstrap() was asked for core's CPU path after "
                "comfy.model_management was imported; core has already picked "
                "its device. Call bootstrap() before the first comfy import.")
        comfy.cli_args.args.cpu = True
    return COMFY


def server_memory_mode() -> bool:
    """Set up ComfyUI's dynamic VRAM layer the way `main.py` does at start.

    Call it after `bootstrap()` and before `comfy.model_management` is
    imported. On the card it is not optional: core's SAM 3D Body loader picks
    half-precision weights with no manual cast, and it is this layer's ops
    that cast a weight to its input at use. Without it core's own DINOv3 path
    stops in `run_keypoint_prompt` on a float input meeting a half weight
    (seen 2026-10-05, on the first card run of
    `bench/compare_sam3d_body_releases.py`, before the ViT-H model was
    reached). Transcribed from `main.py`, the two blocks around
    `comfy_aimdo.control.init` and `init_devices`; returns whether it took.
    With no card visible it does nothing and returns False.
    """
    if not card_visible():
        return False
    from comfy.cli_args import args, enables_dynamic_vram
    if not enables_dynamic_vram():
        return False
    import comfy_aimdo.control
    headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3)
    try:
        comfy_aimdo.control.init(simple_vram_headroom=headroom, nvml_pressure=not args.disable_nvml_pressure)
    except TypeError:
        try:
            comfy_aimdo.control.init(simple_vram_headroom=headroom)
        except TypeError:
            comfy_aimdo.control.init()
    import comfy.memory_management
    import comfy.model_management
    import comfy.model_patcher
    try:
        took = comfy_aimdo.control.init_devices(
            (d.index, int(args.vram_headroom * 1024 ** 3)) for d in comfy.model_management.get_all_torch_devices())
    except TypeError:
        took = comfy_aimdo.control.init_devices(d.index for d in comfy.model_management.get_all_torch_devices())
    if took:
        comfy_aimdo.control.set_log_warning()
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True
    return bool(took)


# ---------------------------------------------------------------------- needs

def needs(what: str, present: bool = False) -> None:
    """Exit 2 with one line unless `present`.

        needs("a CUDA device", card_visible())
        needs("H3_OUTPUT_DIR, where the server writes renders")

    `what` finishes the sentence "nothing graded: needs ...". Say the thing and
    how to supply it; the reader is somebody looking at a sweep table.
    """
    if present:
        return
    print(f"nothing graded: needs {what}")
    raise SystemExit(EXIT_NOT_GRADED)


#: Set by `bench/run_checks.py` in every check it starts. reasoned: a sweep is
#: run to read the state of the repo, so nothing it starts may queue a render
#: on the shared server; a check that would post one asks `in_sweep()` first.
SWEEP_ENV = "H3_CHECK_SWEEP"


def in_sweep() -> bool:
    """Whether `bench/run_checks.py` started this process."""
    import os
    return bool(os.environ.get(SWEEP_ENV))


def home_relative(text: object) -> str:
    """`text` with this machine's home directory written as `~`.

    For a reason that quotes a path from an exception: the line lands in sweep
    logs, and those get copied into notes.
    """
    return str(text).replace(str(Path.home()), "~")


# ---------------------------------------------------------------------- cases

class _Skip(Exception):
    pass


_failed: list[str] = []
_skipped: list[str] = []
_ran = 0


def skip(reason: str) -> None:
    """From inside a case: it could not run, which is neither `ok` nor `FAIL`."""
    raise _Skip(reason)


def case(name: str, fn, *, indent: str = "") -> bool:
    """Run one case and print its line. True when it passed.

    `fn` takes nothing. It passes by returning; a returned string is printed
    after the name. It fails by raising: an AssertionError prints its message,
    anything else prints its type too, since that is a crash, not a verdict.
    """
    global _ran
    _ran += 1
    try:
        detail = fn()
    except _Skip as exc:
        _skipped.append(name)
        print(f"{indent}SKIP  {name}: {exc}")
        return False
    except AssertionError as exc:
        _failed.append(name)
        print(f"{indent}FAIL  {name}: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001 - a crashing case is a failed case
        _failed.append(name)
        print(f"{indent}FAIL  {name}: {type(exc).__name__}: {exc}")
        return False
    print(f"{indent}ok    {name}" + (f": {detail}" if isinstance(detail, str) else ""))
    return True


def refused(target, *words: str) -> str:
    """Assert `target` refuses cleanly with every word in `words` and no Python traceback.

    `target` is either a callable (which must raise `SystemExit` or a `Refused` /
    `ValueError` refusal with a message, never an uncaught crash or a bare exit
    code) or a command list / `subprocess.CompletedProcess` (which must exit
    non-zero with `"Traceback"` absent from stdout + stderr). Returns the refusal
    text.
    """
    import subprocess
    if isinstance(target, (list, tuple)):
        target = subprocess.run([str(x) for x in target], capture_output=True, text=True)
    if isinstance(target, subprocess.CompletedProcess):
        out = ((target.stdout or "") + (target.stderr or "")).strip()
        assert target.returncode != 0, f"refused nothing (exit 0); expected {words}: {out!r}"
        assert out and "Traceback" not in out, f"crashed with a Python traceback or empty refusal: {out!r}"
        assert all(w in out for w in words), f"refused without {words}: {out!r}"
        return out
    try:
        target()
    except SystemExit as exc:
        text = "" if isinstance(exc.code, int) else str(exc.code or "").strip()
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ not in ("Refused", "ValueError"):
            raise AssertionError(f"crashed with {type(exc).__name__}: {exc}") from exc
        text = str(exc).strip()
    else:
        raise AssertionError(f"refused nothing; expected {words}")
    assert text and "Traceback" not in text, f"refusal text empty or holds a traceback: {text!r}"
    assert all(w in text for w in words), f"refused without {words}: {text!r}"
    return text


def finish() -> int:
    """Print the summary and return the exit code. `sys.exit(finish())`.

    A check that registered no case at all returns 2: an empty run proves
    nothing and must not read as a pass.
    """
    print()
    if _failed:
        print(f"{len(_failed)} of {_ran} case(s) FAILED: {', '.join(_failed)}")
        return EXIT_FAIL
    if _skipped:
        print(f"{len(_skipped)} of {_ran} case(s) not graded: {', '.join(_skipped)}")
        return EXIT_NOT_GRADED
    if not _ran:
        print("nothing graded: no case ran")
        return EXIT_NOT_GRADED
    print(f"all {_ran} case(s) ok")
    return EXIT_OK
