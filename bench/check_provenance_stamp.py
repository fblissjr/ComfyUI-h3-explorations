#!/usr/bin/env python3
"""Check that `provenance.py` records the knobs that actually ran.

Why this exists. On 2026-08-16 `SOL_CLOSURE_KEYS` was found to be written in
the **Triton** vocabulary, months after the graphs migrated to the CUDA node.
It was wrong in both directions at once:

  - it asked for `int8_qk`, `use_tma` and `int8_pv`, which the CUDA node does
    not have, so every render recorded them as "not detected" forever;
  - it omitted `routed_cap_percent`, `centroid_tail` and `reuse_qkv_memory`,
    which do run and were therefore absent from the record entirely.

**That is the worst failure this node can have.** A missing field reads as
missing. A field that says "not detected" for a knob that cannot exist reads
exactly like a knob that was switched off, and it would survive into any
postmortem that trusted the stamp. `centroid_tail` is the one that stings: it
has a live A/B with a deadline on it and no stamped render says how it was set.

The same bug hit `builds.sol_attn`, which recorded the *Triton pack's* git HEAD
on renders that ran the CUDA kernel -- an answer, confidently, about a pack
that did not run.

So this check pins the stamp against the node it claims to describe.

What each case claims, i.e. what breaks if it is deleted:

  no_phantom_keys
      Every name in `SOL_CLOSURE_KEYS` is a real parameter of the CUDA node's
      `make_override`. Without this, a key can name a knob that does not exist
      and the stamp reports "not detected" for it on every render, which is
      indistinguishable from a knob that was off. This is the half that was
      wrong for months.

  no_missing_knobs
      Every parameter of `make_override` is in `SOL_CLOSURE_KEYS` (except
      `previous`, which is the chain and not a setting). Without this, a knob
      that genuinely controls the kernel is simply absent from the record and
      nobody notices, because absence of a key looks like nothing at all.

  closure_is_read_not_declared
      **The control that matters.** Two overrides built from the real
      `make_override`, differing in exactly one CUDA-only knob, must produce
      DIFFERENT recorded values. Presence of a key in the output proves the
      key was listed; only a value that MOVES proves it was read. The old
      version passed "is the field there" and failed this.

  absent_is_not_broken
      No override installed must record `state="absent"`, not `"broken"` and
      not an exception. If a graph is run without an override or with Sol bypassed,
      "correctly absent" is handled cleanly.

  version_bump_has_no_consumer
      Records, and re-checks, that NOTHING reads `stamp_schema_version`.
      `STAMP_SCHEMA_VERSION` went 1 -> 2 with the key change, which is correct
      bookkeeping and currently protects nobody: there is no reader to be
      version-aware. This case fails the day a consumer appears without
      handling the version, which is the only moment the bump means anything.

Exit codes: 0 all passed, 1 a case failed, 2 nothing could be checked (the
CUDA node or ComfyUI is not importable).

No GPU, no model, no server. Runs in about a second.

    python bench/check_provenance_stamp.py
"""

from __future__ import annotations

import importlib.util as _ilu
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent

# ComfyUI's root goes FIRST. This repo has its own `nodes.py`, and a bare
# `import nodes` from inside comfy must find ComfyUI's -- the trap
# `docs/comfy_notes.md` records as costing three separate debugging rounds.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(COMFY))


def _load(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    try:
        prov = _load("_h3_provenance_probe", REPO / "provenance.py")
        # The LIVE node, not the pristine vendored reference: this grades
        # what `make_override` actually closes over, and the vendored file
        # has not been the running node since 2026-08-30.
        from _live_sol import live_sol
        sol = live_sol()
    except Exception as exc:  # noqa: BLE001
        print(f"could not import provenance.py / the CUDA Sol node: "
              f"{type(exc).__name__}: {exc}")
        return 2

    failures: list[str] = []

    def report(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<34s} {detail}")
        if not ok:
            failures.append(name)

    keys = tuple(prov.SOL_CLOSURE_KEYS)
    params = inspect.signature(sol.make_override).parameters
    # `previous` chains a prior override; it is plumbing, not a setting.
    knobs = {n for n in params if n != "previous"}

    print(f"provenance stamp v{prov.STAMP_SCHEMA_VERSION} against "
          f"{sol.make_override.__module__}.make_override\n")

    phantom = [k for k in keys if k not in params]
    report("no_phantom_keys", not phantom,
           f"{len(keys)} keys" if not phantom else f"not on the node: {phantom}")

    missing = sorted(knobs - set(keys))
    report("no_missing_knobs", not missing,
           f"{len(knobs)} knobs covered" if not missing else f"unrecorded: {missing}")

    # --- the control: a value that moves, not a key that exists -------------
    # Pick a knob that is BOTH a real parameter and currently recorded. If the
    # key list regresses to the Triton vocabulary this finds nothing to flip
    # and says so, rather than passing vacuously.
    # Updated 2026-08-31: `centroid_tail`, `reuse_qkv_memory` and
    # `routed_cap_percent` are gone from the live node, so a list naming only
    # those would find nothing to flip and report a vacuous pass -- which is
    # the failure this list was written to avoid, arriving by the other route.
    flippable = [k for k in ("tail", "topk_ratio", "tau")
                 if k in params and k in keys]
    if not flippable:
        report("closure_is_read_not_declared", False,
               "no CUDA knob is both a real parameter and recorded -- "
               "nothing to flip, so this case cannot prove anything")
    else:
        knob = flippable[0]
        base = params[knob].default
        other = (not base) if isinstance(base, bool) else (
            (base or 0) + 1 if isinstance(base, (int, float)) else base)
        a = sol.make_override(**{knob: base})
        b = sol.make_override(**{knob: other})
        sa = prov._sol_state({"optimized_attention_override": a}, None)
        sb = prov._sol_state({"optimized_attention_override": b}, None)
        va, vb = sa["closure"].get(knob), sb["closure"].get(knob)
        moved = va != vb and prov.NOT_DETECTED not in (va, vb)
        report("closure_is_read_not_declared", moved,
               f"{knob}: {va!r} -> {vb!r}" if moved
               else f"{knob} did not move ({va!r} vs {vb!r})")

    # --- absent is a state, not a failure ----------------------------------
    try:
        st = prov._sol_state({}, None)
        report("absent_is_not_broken", st.get("state") == "absent",
               f"state={st.get('state')!r}")
    except Exception as exc:  # noqa: BLE001
        report("absent_is_not_broken", False, f"raised {type(exc).__name__}: {exc}")

    # --- the version bump protects nobody yet, and that is the claim -------
    readers = []
    for path in REPO.rglob("*.py"):
        if path.name in ("provenance.py", Path(__file__).name):
            continue
        if "coderef" in path.parts or "internal" in path.parts:
            continue
        try:
            if "stamp_schema_version" in path.read_text():
                readers.append(str(path.relative_to(REPO)))
        except Exception:  # noqa: BLE001, S112
            continue
    report("version_bump_has_no_consumer", not readers,
           "nothing reads the stamp, so every bump so far protects a future "
           "reader only"
           if not readers else
           f"a consumer appeared and must handle the version: {readers}")

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}")
        return 1
    print("\nall ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
