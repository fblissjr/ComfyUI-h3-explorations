#!/usr/bin/env python3
"""Check `bench/masked_code_map.py` AST symbol extraction, dependency/call/data/step chains, and code-map gates.

Runs without CUDA, model weights, or server, and works both in the full repository
(where `docs/masked_v2v/CODE_MAP_INDEX.md` is on disk) and inside `bench/break_checks.py`
temporary copies (where `docs/` is omitted by `LEFT_OUT`).
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from _lib import case, finish  # noqa: E402
import masked_code_map as mcm  # noqa: E402


def check_live_ast_symbols_and_constants() -> None:
    """Verify AST symbol coordinates, digests, tuple unpacking, KEY_CONSTANTS, and CANNOT_SEE_CONSTANTS."""
    all_paths = [p for _, p, _ in mcm.CORE_FILES] + [p for p, _ in mcm.SUPPORT_FILES]
    assert len(all_paths) == 44, f"expected 44 masked-lane files, got {len(all_paths)}"
    for rel_path in all_paths:
        total_lines, syms = mcm.parse_file_symbols(rel_path)
        assert total_lines > 0, f"{rel_path}: total_lines={total_lines}"
        assert syms, f"{rel_path}: parsed 0 symbols"
        for sym, (start, end, digest) in syms.items():
            assert 1 <= start <= end <= total_lines, f"{rel_path}::{sym} bad range L{start}-L{end} / {total_lines}"
            assert len(digest) == 8 and all(c in "0123456789abcdef" for c in digest), (
                f"{rel_path}::{sym} bad digest {digest!r}"
            )

    # Tuple-unpacked constants in h3_rules.py, video_mask.py, bench/strangers.py
    _, h3_syms = mcm.parse_file_symbols("h3_rules.py")
    assert "SHORTEST_CHAINED" in h3_syms and "MAX_LENGTH" in h3_syms, "h3_rules.py tuple constants missing"
    _, vm_syms = mcm.parse_file_symbols("video_mask.py")
    assert "HELD_EDGE_EMPTY_RUN" in vm_syms and "HELD_EDGE_LEAST" in vm_syms, "video_mask.py tuple constants missing"
    _, str_syms = mcm.parse_file_symbols("bench/strangers.py")
    assert "NO_CORE" in str_syms and "NO_OTHER" in str_syms, "bench/strangers.py tuple constants missing"

    # Key constants & cannot_see constants resolve cleanly
    assert len(set(mcm.KEY_CONSTANTS)) == 20 and ("h3_rules.py", "MAX_LENGTH") in mcm.KEY_CONSTANTS, (
        "KEY_CONSTANTS must hold all 20 distinct constants including h3_rules.py::MAX_LENGTH"
    )
    for rel_path, const_name in mcm.KEY_CONSTANTS:
        lineno, val = mcm.extract_constant_repr(rel_path, const_name)
        assert lineno >= 1, f"KEY_CONSTANTS {rel_path}::{const_name} bad line L{lineno}"
        assert val and val != "?", f"KEY_CONSTANTS {rel_path}::{const_name} empty repr"
    assert mcm.extract_constant_repr("h3_rules.py", "MAX_LENGTH")[1] == "362"
    assert mcm.extract_constant_repr("video_mask.py", "HELD_EDGE_EMPTY_RUN")[1] == "max(FRAME_PER_TOKEN)"
    assert mcm.extract_constant_repr("video_mask.py", "HELD_EDGE_LEAST")[1] == "0.25"

    for rel_path, const_name, _ in mcm.CANNOT_SEE_CONSTANTS:
        start, end, items = mcm.extract_cannot_see_constant(rel_path, const_name)
        assert 1 <= start <= end, f"CANNOT_SEE_CONSTANTS {rel_path}::{const_name} bad lines L{start}-L{end}"
        assert items, f"CANNOT_SEE_CONSTANTS {rel_path}::{const_name} unexpectedly empty: {items!r}"


def check_dependency_and_call_graphs() -> None:
    """Verify module-level bidirectional dependency edges and function-level AST call graph."""
    dep_graph = mcm.compute_dependency_graph()
    assert "bench/load_chain.py" in dep_graph["bench/shot_to_load.py"]["imports"], (
        "bench/shot_to_load.py must import bench/load_chain.py"
    )
    assert "bench/shot_to_load.py" in dep_graph["bench/load_chain.py"]["imported_by"], (
        "bench/load_chain.py must be imported_by bench/shot_to_load.py"
    )
    assert "video_mask.py" in dep_graph["audio_freeze_song.py"]["imports"], (
        "audio_freeze_song.py must import video_mask.py"
    )
    assert "video_mask.py" in dep_graph["bench/measure_blocks.py"]["dynamic"], (
        "bench/measure_blocks.py must dynamically load video_mask.py"
    )
    assert "bench/pose_along_shot.py" in dep_graph["bench/load_chain.py"]["dynamic"], (
        "bench/load_chain.py must invoke bench/pose_along_shot.py dynamically"
    )
    assert "bench/load_chain.py" in dep_graph["bench/pose_along_shot.py"]["dynamic_by"], (
        "bench/pose_along_shot.py must record bench/load_chain.py in dynamic_by"
    )

    callees, callers, coords = mcm.compute_call_graph()
    ht_key = ("bench/head_tube_parts.py", "sam31_head_tube")
    assert ht_key in coords, "sam31_head_tube missing from call graph coords"
    ht_callers = {f"{p}::{s}" for p, s, _ in callers.get(ht_key, [])}
    assert "bench/head_tube_parts.py::head_tube_for_capture" in ht_callers, (
        f"expected head_tube_for_capture in callers of sam31_head_tube, got {ht_callers}"
    )
    ht_callees = {f"{p}::{s}" for p, s, _ in callees.get(ht_key, [])}
    assert "bench/head_tube_parts.py::bridge_short_tube_dips" in ht_callees, (
        f"expected bridge_short_tube_dips in callees of sam31_head_tube, got {ht_callees}"
    )

    uh_key = ("bench/capture_masked_run.py", "unmasked_head_hair_context")
    uh_callers = {f"{p}::{s}" for p, s, _ in callers.get(uh_key, [])}
    for expected_caller in (
        "bench/capture_masked_run.py::flag_her_hair",
        "bench/capture_masked_run.py::flag_lent",
        "bench/capture_masked_run.py::lays_nothing",
    ):
        assert expected_caller in uh_callers, (
            f"expected {expected_caller} in callers of unmasked_head_hair_context, got {uh_callers}"
        )


def check_data_chain_and_step_chain() -> None:
    """Verify both-ends artifact writer/reader AST scanning and CLI step -> subprocess chain."""
    pg_hits = mcm.scan_ast_data_hits("part_grades.json")
    pg_writers = {f"{rel}::{fn}" for role, rel, fn, _, _, _, _ in pg_hits if role == "WRITE"}
    pg_readers = {f"{rel}::{fn}" for role, rel, fn, _, _, _, _ in pg_hits if role == "READ"}
    assert "bench/capture_masked_run.py::_files" in pg_writers, (
        f"expected capture_masked_run.py::_files as writer of part_grades.json, got {pg_writers}"
    )
    for expected_reader in (
        "bench/capture_masked_run.py::unmasked_head_hair_context",
        "bench/flags_before_render.py::unmasked_frame_diagnostics_of",
        "bench/frame_table.py::build_frame_table",
        "bench/frame_coverage.py::Capture.grades",
    ):
        assert expected_reader in pg_readers, (
            f"expected {expected_reader} as reader of part_grades.json, got {pg_readers}"
        )

    reg_hits = mcm.scan_ast_data_hits("region.npz")
    reg_writers = {f"{rel}::{fn}" for role, rel, fn, _, _, _, _ in reg_hits if role == "WRITE"}
    assert "bench/capture_masked_run.py::_files" in reg_writers, (
        f"expected _files as writer of region.npz, got {reg_writers}"
    )

    steps = mcm.compute_step_chain()
    by_step = {(rel, step_name): targets for rel, step_name, _, _, _, _, targets in steps}
    assert ("bench/shot_to_load.py", "preview") in by_step
    assert "bench/masked_job.py" in by_step[("bench/shot_to_load.py", "preview")], (
        "preview step must list bench/masked_job.py"
    )
    assert ("bench/load_chain.py", "first") in by_step
    assert "bench/capture_masked_run.py" in by_step[("bench/load_chain.py", "first")], (
        "first step must list bench/capture_masked_run.py"
    )
    assert ("bench/load_chain.py", "no-look") in by_step
    assert "bench/flags_before_render.py::unresolved_hair_in_hole_of" in by_step[("bench/load_chain.py", "no-look")], (
        "no-look step must list flags_before_render.py::unresolved_hair_in_hole_of"
    )


def check_generated_index_and_synthetic_controls() -> None:
    """Verify generated index invariants and synthetic negative controls on `check_all`."""
    md = mcm.generate_index_markdown()
    raw = md.encode("utf-8")
    assert len(raw) <= mcm.MAX_FILE_BYTES, f"generated index {len(raw)} bytes > {mcm.MAX_FILE_BYTES}"
    max_line = max(len(line) for line in md.splitlines())
    assert max_line <= mcm.MAX_LINE_CHARS, f"generated index max_line {max_line} > {mcm.MAX_LINE_CHARS}"
    for section in (
        "## Part 3 — Subcommand & Function Jump Index for `read_file` Calls",
        "## Part 2d — Declared `cannot_see` Boundaries Across All Tools",
        "## Part 2e — Live AST Constants & Module Dependency Chain",
        "### 2e.1 Core Masked-Lane Constants (Extracted from AST)",
        "### 2e.2 Module Dependency Graph (AST Static Imports & Dynamic `_pack`/CLI Edges)",
        "### 2e.3 Pipeline Step Dispatch & Data-Chain CLI Queries (Extracted from AST)",
    ):
        assert section in md, f"missing section {section!r} in generated index"

    # Synthetic controls in a temporary directory (works even when docs/ is omitted by break_checks.py)
    with tempfile.TemporaryDirectory(prefix="h3_code_map_test_") as tmp:
        tdir = Path(tmp)
        idx_p = tdir / "CODE_MAP_INDEX.md"
        stages_p = tdir / "code_map_stages.md"
        contracts_p = tdir / "code_map_contracts.md"
        plan_p = tdir / "2026-10-26_code_map_and_simplification_plan.md"
        docs_tuple = (idx_p, stages_p, contracts_p, plan_p)

        idx_p.write_text(md, encoding="utf-8")
        stages_p.write_text("# Stages\n- Calls `bench/shot_to_load.py::step_preview`.\n", encoding="utf-8")
        contracts_p.write_text("# Contracts\n- Calls `video_mask.py::lay_window`.\n", encoding="utf-8")
        plan_p.write_text("# Plan\n- Calls `bench/head_tube_parts.py::sam31_head_tube`.\n", encoding="utf-8")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mcm.check_all(index_md=idx_p, code_map_docs=docs_tuple, expected_md=md)
        assert rc == 0, f"expected clean synthetic docs to pass check_all, got rc={rc}:\n{buf.getvalue()}"

        # Negative Control 1: Stale CODE_MAP_INDEX.md
        idx_p.write_text(md + "\n<!-- drift -->\n", encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mcm.check_all(index_md=idx_p, code_map_docs=docs_tuple, expected_md=md)
        assert rc == 1 and "out of date with live AST extraction" in buf.getvalue(), (
            "expected stale CODE_MAP_INDEX.md to fail check_all"
        )
        idx_p.write_text(md, encoding="utf-8")

        # Negative Control 2: Missing symbol citation
        stages_p.write_text("# Stages\n- Calls `bench/shot_to_load.py::no_such_symbol_xyz`.\n", encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mcm.check_all(index_md=idx_p, code_map_docs=docs_tuple, expected_md=md)
        assert rc == 1 and "does not exist in `bench/shot_to_load.py`'s AST" in buf.getvalue(), (
            "expected missing AST symbol citation to fail check_all"
        )
        stages_p.write_text("# Stages\n- Calls `bench/shot_to_load.py::step_preview`.\n", encoding="utf-8")

        # Negative Control 3: Line over MAX_LINE_CHARS
        contracts_p.write_text("# Contracts\n" + ("x" * (mcm.MAX_LINE_CHARS + 10)) + "\n", encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mcm.check_all(index_md=idx_p, code_map_docs=docs_tuple, expected_md=md)
        assert rc == 1 and "MAX_LINE_CHARS" in buf.getvalue(), (
            "expected over-length line to fail check_all"
        )
        contracts_p.write_text("# Contracts\n- Calls `video_mask.py::lay_window`.\n", encoding="utf-8")

        # Negative Control 4: File over MAX_FILE_BYTES
        plan_p.write_text("# Plan\n" + ("short line\n" * ((mcm.MAX_FILE_BYTES // 10) + 100)), encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mcm.check_all(index_md=idx_p, code_map_docs=docs_tuple, expected_md=md)
        assert rc == 1 and "MAX_FILE_BYTES" in buf.getvalue(), (
            "expected oversized file to fail check_all"
        )

        # Negative Control 5: Leaked POSIX home path (constructed dynamically so check_no_owner_paths stays green)
        fake_home = "/" + "home" + "/testowner/secret.mp4"
        plan_p.write_text(f"# Plan\n- Path: {fake_home}\n", encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mcm.check_all(index_md=idx_p, code_map_docs=docs_tuple, expected_md=md)
        assert rc == 1 and "contains a home-directory path" in buf.getvalue(), (
            "expected home-directory path leak to fail check_all"
        )

    # Live repo check when docs/ is present on disk
    if mcm.INDEX_MD.is_file():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mcm.check_all(expected_md=md)
        assert rc == 0, f"live check_all() failed:\n{buf.getvalue()}"


def main() -> int:
    case("live_ast_symbols_and_constants", check_live_ast_symbols_and_constants)
    case("dependency_and_call_graphs", check_dependency_and_call_graphs)
    case("data_chain_and_step_chain", check_data_chain_and_step_chain)
    case("generated_index_and_synthetic_controls", check_generated_index_and_synthetic_controls)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
