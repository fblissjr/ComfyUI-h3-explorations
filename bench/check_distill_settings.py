#!/usr/bin/env python3
"""Check every turbo LoRA is loaded at the shift and step count it was distilled at.

The three FL2VA turbo LoRAs do not share a schedule. Two were distilled at
video shift 12, one at 6:

    FL2VA Turbo 4-step v0.1     544p mixed aspect   12 / 3   4 steps
    FL2VA Turbo 8-step v1.0     544p mixed aspect   12 / 3   8 or 4 steps
    FL2VA Turbo 4-step v1.0     768p (1344x768)      6 / 3   4 steps

A turbo LoRA inherits the sampler's shift; it does not carry its own. So
loading the 768p one into a graph whose `MiniMaxH3SigmaShift` still reads
12/3 samples it off a schedule it never saw, and **nothing errors**. The
render completes and looks plausibly wrong. Steps move with the LoRA for the
same reason: 16 is a base-model number.

Claims, i.e. what breaks if a case is deleted:
  vendor table agrees   both the shifts AND the recommended step counts in
                        LEGAL are graded against the vendor's own README in
                        coderef/, not against themselves. Grading only the
                        shifts would leave the step sets self-checked, which
                        is the trap `check_reference_fit.py` learned the hard
                        way. Absent coderef/ this SKIPS and the script exits
                        2, not 0, because a control that is quietly not run
                        reads as a pass to anything keying on the exit code
  config is consistent   h3_config.py's TURBO_LORA / TURBO_STEPS /
                        TURBO_SHIFT triple is one of the legal rows. This is
                        what the generator writes into every graph, so it is
                        the single upstream point where a mismatch is born
  graphs are consistent  EVERY shipped graph, not just the turbo ones. A
                        graph with a turbo LoRA must match that LoRA's row; a
                        graph without one must sit at the base checkpoint's
                        own 12/3. Skipping the base graphs is how "every
                        shipped graph" quietly becomes "the two with a LoRA".
                        The UI and API forms are then paired and compared:
                        they are generated separately and have already
                        diverged once (the ref `_api` graphs hardcode length),
                        so checking one does not check the other
  unknown lora is caught a lora whose filename matches no known row fails
                        rather than passing unexamined. Includes the prefix
                        trap: `turbo_8step_v1.0_768p` must NOT resolve to the
                        12/3 `turbo_8step_v1.0` row just because that name is
                        a prefix of it. Names are parsed structurally, not by
                        substring, for exactly this reason

No CUDA, no model, no ComfyUI import. Reads JSON and a config module.

Exit codes: 0 all cases passed, 1 a case failed, 2 passed but a control was
skipped (coderef/ absent).

    python bench/check_distill_settings.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workflows"))

WORKFLOWS = REPO / "workflows"
VENDOR_README = REPO / "coderef" / "Minimax-H3-Turbo" / "README.md"


class Row(NamedTuple):
    shift_video: float
    shift_audio: float
    steps: frozenset[int]


class Found(NamedTuple):
    loras: list[str]
    shift: tuple[float, float] | None
    steps: int | None


# Keyed by the distinguishing fragment of the ComfyUI filename.
LEGAL: dict[str, Row] = {
    "turbo_4step_v0.1": Row(12.0, 3.0, frozenset({4})),
    "turbo_8step_v1.0": Row(12.0, 3.0, frozenset({8, 4})),
    "turbo_4step_v1.0_768p": Row(6.0, 3.0, frozenset({4})),
}

# The base checkpoint's own training shifts. Every graph that loads no turbo
# LoRA must sit here, which is what makes "every shipped graph" true rather
# than "the two graphs that happen to load a LoRA".
BASE_SHIFT = (12.0, 3.0)

# turbo_<n>step_<version>[_<resolution>] -- parsed structurally, NOT by
# substring. A substring match classifies a hypothetical
# `turbo_8step_v1.0_768p` as the 12/3 `turbo_8step_v1.0` row, because the
# latter is a prefix of the former. That is a 768p checkpoint silently
# inheriting the wrong shift, i.e. exactly the failure this file exists to
# catch, committed by the file itself.
_NAME = re.compile(r"turbo_(\d+)step_(v\d+\.\d+)(?:_(\d+p))?", re.IGNORECASE)


def classify(lora_name):
    """Which legal row a lora filename belongs to, or None if unrecognised."""
    m = _NAME.search(lora_name.replace("-", "_"))
    if not m:
        return None
    steps, version, res = m.groups()
    key = f"turbo_{steps}step_{version.lower()}"
    if res:
        key += f"_{res.lower()}"
    return key if key in LEGAL else None


def is_turbo(lora_name):
    return "turbo" in lora_name.lower()


# --------------------------------------------------------------------------
# graph readers -- the two shipped forms store the same graph differently
# --------------------------------------------------------------------------

def _literal(value):
    """An API-form input is either a literal or a `[node_id, slot]` link.

    A linked widget is not a value this file can grade, and coercing one with
    `float()` raises TypeError -- which surfaces as a FAIL on a graph that is
    actually fine. Return None so the caller reports "could not read" instead.
    """
    return None if isinstance(value, list) else value


def read_api(doc) -> Found:
    """API form: {node_id: {class_type, inputs{}}}."""
    loras: list[str] = []
    shift: tuple[float, float] | None = None
    steps: int | None = None
    for node in doc.values():
        ct, inp = node.get("class_type"), node.get("inputs", {})
        if ct == "LoraLoaderModelOnly":
            loras.append(inp.get("lora_name", ""))
        elif ct == "MiniMaxH3SigmaShift":
            sv, sa = _literal(inp.get("shift_video")), _literal(inp.get("shift_audio"))
            shift = None if sv is None or sa is None else (float(sv), float(sa))
        elif ct == "BasicScheduler":
            n = _literal(inp.get("steps"))
            steps = None if n is None else int(n)
    return Found(loras, shift, steps)


def read_ui(doc) -> Found:
    """UI form: widgets_values positionally. LoraLoaderModelOnly is
    [name, strength]; MiniMaxH3SigmaShift is [video, audio]; BasicScheduler
    is [scheduler, steps, denoise]."""
    loras: list[str] = []
    shift: tuple[float, float] | None = None
    steps: int | None = None
    for node in doc.get("nodes", []):
        t, w = node.get("type"), node.get("widgets_values") or []
        if t == "LoraLoaderModelOnly" and w:
            loras.append(w[0])
        elif t == "MiniMaxH3SigmaShift" and len(w) >= 2:
            shift = (float(w[0]), float(w[1]))
        elif t == "BasicScheduler" and len(w) >= 2:
            steps = int(w[1])
    return Found(loras, shift, steps)


def parse_vendor_table(text):
    """Pull (shift_video, shift_audio, recommended_steps) out of the vendor README.

    Their table's centred cells run: tasks, training resolution, training
    shifts ("12 / 3"), distillation steps, recommended inference steps
    ("8 / 4" for the 8-step, "4" for the others). Rows are keyed by the
    checkpoint filename in the row's links, the only stable handle across
    their formatting.

    Both the shift pair and the recommended steps are read. Grading only the
    shifts would leave `LEGAL`'s step sets checked against themselves, which
    is the self-consistency trap this control exists to avoid.
    """
    rows = {}
    for key in LEGAL:
        m = re.search(re.escape(key), text)
        if not m:
            continue
        start = text.rfind("<tr>", 0, m.start())
        end = text.find("</tr>", m.end())
        if start < 0 or end < 0:
            continue
        block = text[start:end]
        cells = [c.strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", block, re.S)]
        # strip tags out of each cell so "<sub>" notes do not confuse the digits
        cells = [re.sub(r"<[^>]+>", " ", c).strip() for c in cells]
        shift = next((c for c in cells if re.fullmatch(r"\d+\s*/\s*\d+", c)), None)
        if shift is None:
            continue
        sv, sa = (float(v) for v in shift.split("/"))
        # recommended steps is the last centred cell; "8 / 4" means either
        steps = frozenset(int(v) for v in re.findall(r"\d+", cells[-1]))
        rows[key] = (sv, sa, steps)
    return rows


def main():
    failures, skipped = [], []

    def check(name, fn):
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {exc}")

    print("turbo LoRA shift and step pairing")

    # ---- the table against the vendor's own README -----------------------
    def vendor_table_agrees():
        rows = parse_vendor_table(VENDOR_README.read_text(encoding="utf-8"))
        if not rows:
            raise AssertionError(
                f"parsed no rows from {VENDOR_README}; the vendor reformatted "
                "their table and this control is no longer reading it")
        for key, (sv, sa, steps) in rows.items():
            want = LEGAL[key]
            assert (sv, sa) == (want.shift_video, want.shift_audio), (
                f"{key}: vendor says shift {sv}/{sa}, our table says "
                f"{want.shift_video}/{want.shift_audio}")
            assert steps == want.steps, (
                f"{key}: vendor recommends steps {sorted(steps)}, our table "
                f"says {sorted(want.steps)}")
        missing = set(LEGAL) - set(rows)
        assert not missing, f"vendor README has no row for {sorted(missing)}"

    if not VENDOR_README.exists():
        skipped.append("vendor table agrees")
        print(f"  SKIP  vendor table agrees: {VENDOR_README} not present "
              "(coderef/ is gitignored). The remaining cases only prove "
              "self-consistency.")
    else:
        check("vendor table agrees", vendor_table_agrees)

    # ---- h3_config.py ----------------------------------------------------
    def config_is_consistent():
        import h3_config  # noqa: WPS433
        key = classify(h3_config.TURBO_LORA)
        assert key, f"TURBO_LORA {h3_config.TURBO_LORA!r} matches no known row"
        want = LEGAL[key]
        got = (float(h3_config.TURBO_SHIFT["shift_video"]),
               float(h3_config.TURBO_SHIFT["shift_audio"]))
        assert got == (want.shift_video, want.shift_audio), (
            f"{key}: config shift {got}, distilled at "
            f"({want.shift_video}, {want.shift_audio})")
        assert h3_config.TURBO_STEPS in want.steps, (
            f"{key}: config steps {h3_config.TURBO_STEPS}, "
            f"distilled for {sorted(want.steps)}")

    check("h3_config turbo triple is legal", config_is_consistent)

    # ---- every shipped graph, both forms ---------------------------------
    def graphs_are_consistent():
        turbo_graphs, base_graphs = {}, {}
        for path in sorted(WORKFLOWS.glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            found = read_api(doc) if "nodes" not in doc else read_ui(doc)
            turbo = [l for l in found.loras if is_turbo(l)]

            if not turbo:
                # A base graph is still policed: it must sit at the base
                # checkpoint's own training shifts. Skipping it is how
                # "every shipped graph" quietly became "the two with a LoRA".
                if found.shift is not None:
                    base_graphs[path.name] = found
                    assert found.shift == BASE_SHIFT, (
                        f"{path.name}: no turbo LoRA, so it must sit at the base "
                        f"shift {BASE_SHIFT[0]}/{BASE_SHIFT[1]}, "
                        f"has {found.shift[0]}/{found.shift[1]}")
                continue

            turbo_graphs[path.name] = (found, turbo)
            for lora in turbo:
                key = classify(lora)
                assert key, (
                    f"{path.name}: lora {lora!r} matches no known row. A "
                    "checkpoint we do not know the shift for must fail here "
                    "rather than inherit someone else's.")
                want = LEGAL[key]
                assert found.shift is not None, (
                    f"{path.name}: loads {key} but has no MiniMaxH3SigmaShift; "
                    "there is nowhere to notice the shift is wrong")
                assert found.shift == (want.shift_video, want.shift_audio), (
                    f"{path.name}: {key} wants shift "
                    f"{want.shift_video}/{want.shift_audio}, "
                    f"graph has {found.shift[0]}/{found.shift[1]}")
                assert found.steps is not None, (
                    f"{path.name}: loads {key} but no BasicScheduler step count "
                    "could be read; steps move with the LoRA and nothing here "
                    "would notice")
                assert found.steps in want.steps, (
                    f"{path.name}: {key} wants steps {sorted(want.steps)}, "
                    f"graph has {found.steps}")

        assert turbo_graphs, "no shipped graph loads a turbo LoRA; this check saw nothing"
        assert base_graphs, "no shipped base graph was examined; the base arm is unpoliced"

        # The UI and API forms are generated separately and have already
        # diverged once (the ref _api graphs hardcode `length`). Checking one
        # is not checking the other, so pair them explicitly.
        paired = 0
        for name, (found, turbo) in turbo_graphs.items():
            if name.endswith("_api.json"):
                continue
            sibling = name[:-5] + "_api.json"
            assert sibling in turbo_graphs, (
                f"{name} loads a turbo LoRA but {sibling} does not; the two "
                "forms of one graph disagree about the arm they run")
            other, other_turbo = turbo_graphs[sibling]
            assert (found.shift, found.steps, sorted(turbo)) == \
                   (other.shift, other.steps, sorted(other_turbo)), (
                f"{name} and {sibling} disagree: "
                f"{found.shift}/{found.steps}/{sorted(turbo)} vs "
                f"{other.shift}/{other.steps}/{sorted(other_turbo)}")
            paired += 1
        assert paired, "no UI/API pair was compared"
        print(f"        ({len(turbo_graphs)} turbo, {len(base_graphs)} base, "
              f"{paired} UI/API pair(s))")

    check("every shipped graph sits at the right shift and steps", graphs_are_consistent)

    # ---- an unknown checkpoint must not pass unexamined -------------------
    def unknown_lora_is_caught():
        assert classify("h3/minimax_h3_fl2v_turbo_16step_v2.0_comfy.safetensors") is None, (
            "a checkpoint we have never seen classified as a known row; "
            "it would inherit that row's shift silently")
        assert classify("h3/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors") \
            == "turbo_4step_v1.0_768p", "768p must not be shadowed by the 4step_v1.0 row"
        # The prefix trap: a substring match reads this as `turbo_8step_v1.0`
        # and hands it 12/3, when a 768p variant would be distilled at 6.
        assert classify("h3/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors") \
            is None, ("a 768p variant of a known row inherited the non-768p "
                      "row's shift; that is the exact silent failure this "
                      "file exists to catch")
        assert classify("h3/minimax_h3_fl2v_turbo_4step_v9.9.safetensors") is None, (
            "an unknown version of a known step count was classified")

    check("an unknown turbo checkpoint is not classified", unknown_lora_is_caught)

    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    if skipped:
        # Exit 2, not 0. The whole point of the vendor case is that our table
        # is not graded against itself; if it did not run, a caller keying on
        # the exit code must be able to tell that apart from a clean pass.
        print(f"\n{len(skipped)} case(s) SKIPPED: {', '.join(skipped)}. "
              "Exit 2: the remaining cases only prove self-consistency.")
        return 2
    print("\nall ok -- every graph sits at the shift and steps its arm was distilled at")
    return 0


if __name__ == "__main__":
    sys.exit(main())
