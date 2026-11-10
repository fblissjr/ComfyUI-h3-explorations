#!/usr/bin/env python3
"""Build, inspect, meter, and grade MiniMax H3 control vectors.

What it checks: builds `.safetensors` control vectors from contrastive prompt
pair files (`prompt_bank/control_vectors/*.toml`), inspects saved vector files
and compatibility recipes, meters bank prompts along a control vector without
rendering (E2.8), and grades activation/prediction movement against `flip_null`,
`random_null`, the `" ."` period edit, the prompt-edit baseline, and a
length-matched padding control (E2.1-E2.7).
"""

from __future__ import annotations

from pathlib import Path
import sys

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
if __name__ == "__main__":
    import importlib.util as _ilu
    import os as _os

    if _ilu.find_spec("torch") is None:
        _venv_py = REPO_ROOT.parent.parent / ".venv" / "bin" / "python"
        if _venv_py.is_file() and Path(sys.executable).resolve() != _venv_py.resolve():
            _os.execv(str(_venv_py), [str(_venv_py), *sys.argv])

# Ensure REPO_ROOT precedes BENCH_DIR on sys.path so `import control_vectors`
# resolves to the root library module (`REPO_ROOT/control_vectors.py`), while
# keeping BENCH_DIR on sys.path so `bench/_lib/__init__.py` can import `_lib.*`.
sys.path = [p for p in sys.path if Path(p or ".").resolve() not in (REPO_ROOT, BENCH_DIR)]
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(REPO_ROOT))

# Guard against callers that inserted `bench/` ahead of `REPO_ROOT` on `sys.path`
# and ran `import control_vectors` (which binds `sys.modules["control_vectors"]`
# to `bench/control_vectors.py` before `REPO_ROOT/control_vectors.py` is loaded).
_shadowed_as_root = (
    __name__ == "control_vectors"
    and getattr(sys.modules.get("control_vectors"), "__file__", None) == __file__
)
if _shadowed_as_root:
    import importlib.util as _root_ilu

    _this_shadow_mod = sys.modules["control_vectors"]
    sys.modules["bench.control_vectors"] = _this_shadow_mod
    _root_spec = _root_ilu.spec_from_file_location(
        "control_vectors", REPO_ROOT / "control_vectors.py"
    )
    assert _root_spec is not None and _root_spec.loader is not None
    _root_cv_mod = _root_ilu.module_from_spec(_root_spec)
    sys.modules["control_vectors"] = _root_cv_mod
    _root_spec.loader.exec_module(_root_cv_mod)
    for _k, _v in vars(_root_cv_mod).items():
        if not _k.startswith("__"):
            globals()[_k] = _v

from bench._lib import bootstrap  # noqa: E402

bootstrap()

import folder_paths  # noqa: E402,F401
import torch  # noqa: E402,F401

import control_vector_nodes as cv_nodes  # noqa: E402,F401
import control_vectors as cv  # noqa: E402,F401
import h3_pair_compiler as pair_compiler  # noqa: E402,F401
import h3_spans  # noqa: E402,F401
import j_lens  # noqa: E402,F401
import substrate  # noqa: E402,F401
from workflows import h3_config  # noqa: E402,F401

from bench.control_vector import (  # noqa: E402,F401
    BENCH_DIR,
    ControlVectorUIState,
    LOOK_FAMILY_IDS,
    PACKAGE_DIR,
    REPO_ROOT,
    SUPPORTED_LLM_PROVIDERS,
    WEB_DIR,
    _apply_post_refiner_holds,
    _attach_raw_vector,
    _block_metrics,
    _build_padding_prompt,
    _build_pairset_from_ui_rows,
    _compile_words_or_spec_to_pairset,
    _compose_and_save_sculpted_vector,
    _cosine,
    _decode_image_spec_to_tensor,
    _discover_ui_clips,
    _display_path,
    _extract_system_prompt_text,
    _free_idle_comfy_vram,
    _generate_pairs_with_llm,
    _is_markdown_pairs_text,
    _list_ui_toml_files,
    _list_ui_vectors,
    _load_clip,
    _load_encoded_references,
    _load_image_tensor_list,
    _meter_prompt_tokens,
    _one_forward,
    _pair_set_to_markdown,
    _parse_lora_specs,
    _pred_blocks,
    _preview_markdown_pairs,
    _prompt_token_count,
    _recommended_strength_text,
    _rel_l2,
    _repair_multichange_markdown_pairs,
    _repool_cached_build,
    _resolve_exl3_library_root,
    _resolve_prompt_text,
    _resolve_requested_families,
    _resolve_vector_for_family,
    _sigma_shifted,
    build_parser,
    cmd_build,
    cmd_build_j_lens,
    cmd_from_md,
    cmd_generate_pairs,
    cmd_grade,
    cmd_inspect,
    cmd_make,
    cmd_make_media,
    cmd_rebuild,
    cmd_sculpt,
    cmd_ui,
    cmd_words,
    handle_ui_request,
    main,
    verify_j_lens_bundle,
)
from bench.control_vector import helpers as _helpers  # noqa: E402
import types  # noqa: E402


class _ControlVectorsModule(types.ModuleType):
    """Forwards monkey-patched helper attributes (such as `_load_clip` in `bench/check_control_vectors.py`)
    directly to `bench.control_vector.helpers` so submodules see test overrides."""

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if hasattr(_helpers, name):
            setattr(_helpers, name, value)


_this_mod = sys.modules.get(__name__)
if _this_mod is None:
    for _m in list(sys.modules.values()):
        if getattr(_m, "__file__", None) == __file__:
            _this_mod = _m
            break
if _this_mod is not None:
    _this_mod.__class__ = _ControlVectorsModule
else:
    import inspect as _inspect

    _frame = _inspect.currentframe()
    if _frame is not None:
        # Loaded via spec_from_file_location without sys.modules registration;
        # register in sys.modules under __name__ so __class__ assignment works.
        for _gc_obj in __import__("gc").get_objects():
            if isinstance(_gc_obj, types.ModuleType) and getattr(_gc_obj, "__name__", None) == __name__ and getattr(_gc_obj, "__file__", None) == __file__:
                _gc_obj.__class__ = _ControlVectorsModule
                break

if _shadowed_as_root:
    for _k, _v in list(globals().items()):
        if not _k.startswith("__") and not hasattr(_root_cv_mod, _k):
            setattr(_root_cv_mod, _k, _v)
    _root_cv_mod.__class__ = _ControlVectorsModule


if __name__ == "__main__":
    raise SystemExit(main())
