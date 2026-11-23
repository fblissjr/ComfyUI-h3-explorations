"""Thin backward-compatibility shim aliasing `bench/control_vector/pair_compiler.py`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_pair_compiler():
    impl_key = "bench.control_vector.pair_compiler"
    existing = sys.modules.get(impl_key)
    if existing is not None and hasattr(existing, "compile_words_to_pairset"):
        return existing
    target = Path(__file__).resolve().parent / "bench" / "control_vector" / "pair_compiler.py"
    spec = importlib.util.spec_from_file_location(impl_key, target)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {target}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[impl_key] = mod
    spec.loader.exec_module(mod)
    return mod


_impl = _load_pair_compiler()
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
sys.modules[__name__] = _impl
