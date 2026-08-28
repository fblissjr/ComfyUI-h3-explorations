#!/usr/bin/env python3
"""Check every turbo LoRA is loaded at the shift and step count it was distilled at.

The lightx2v turbo LoRAs do not share a schedule. The 544p ones were
distilled at video shift 12, the 768p ones at 6:

    FL2VA Turbo 4-step v0.1     544p mixed aspect   12 / 3   4 steps
    FL2VA Turbo 8-step v1.0     544p mixed aspect   12 / 3   8 or 4 steps
    FL2VA Turbo 4-step v1.0     768p (1344x768)      6 / 3   4 steps
    Ref2VA Turbo 4-step v0.1    544p mixed aspect   12 / 3   4 steps
    FL2VA Turbo 4-step v0.1 SLA 768p (1344x768)      6 / 3   4 steps

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
                        reads as a pass to anything keying on the exit code.
                        The SLA row is not in that README (it shipped as a
                        separate HF repo), so it is graded against the
                        LightX2V inference config that loads it: shifts read
                        directly, steps as `infer_steps - 1`. That N+1
                        convention is itself checked, not assumed -- every
                        row present in BOTH sources must satisfy it, or the
                        case fails before the SLA row is read
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

# A bare `WORKFLOWS.glob` is non-recursive and quietly stops covering any
# directory `GRAPH_DIRS` routes to -- as `workflows/image/` was from 2026-08-16
# until the single-frame lane was parked on 2026-08-27. See h3_config.GRAPH_DIRS.
from h3_config import (LORA_LOADER_CLASSES, graph_paths, graph_schedule,  # noqa: E402
                        resolve_link, turbo_label)


class Row(NamedTuple):
    shift_video: float
    shift_audio: float
    steps: frozenset[int]


class Found(NamedTuple):
    loras: list[str]
    shift: tuple[float, float] | None
    steps: int | None
    # Read here, graded nowhere in this file. `check_distill_grid.py` imports
    # these readers rather than growing a second graph walk, and the scheduler
    # is the field it needs: a LoRA loaded at the right shift and step count is
    # still off its distillation grid if the scheduler places the steps
    # somewhere else.
    scheduler: str | None = None
    # EVERY shift node, in iteration order. `shift` above keeps the last one
    # for the callers that want a single answer; a split graph carries two
    # (`build_workflows.py::_plain_model_chain` builds node 40 beside node 19)
    # and reading one of two silently grades half the graph.
    shifts: tuple[tuple[float, float], ...] = ()
    # {lora filename: strength_model}. Strength was read by nothing until
    # 2026-08-23: a graph at the right file, shift and steps but the wrong
    # strength is a different arm, and three of four fields staying right is
    # exactly how the fourth drifts unnoticed.
    strengths: dict[str, float] | None = None
    # The `nfe` OVERRIDE a PDD graph sets, or None at its default of 0.
    # Added 2026-08-26 when the node began fusing heads at load for any
    # divisor; repurposed 2026-08-27 when it began deriving the count from
    # `sample_sigmas` instead. It is no longer the evaluation count -- the
    # sampler's `steps` is -- so this exists only to catch a graph that forces
    # one partition while stepping another.
    pdd_nfe: int | None = None


# Keyed by the distinguishing fragment of the ComfyUI filename.
LEGAL: dict[str, Row] = {
    "turbo_4step_v0.1": Row(12.0, 3.0, frozenset({4})),
    "turbo_8step_v1.0": Row(12.0, 3.0, frozenset({8, 4})),
    "turbo_4step_v1.0_768p": Row(6.0, 3.0, frozenset({4})),
    # The SLA release. Parses to this key because its filename is
    # `..._4step_v0.1_768p_sla_...`; the `_768p` keeps it off the 544p v0.1
    # row, which is the prefix trap below working as intended. Not in the
    # Turbo README -- graded against LIGHTX2V_CONFIGS instead.
    "turbo_4step_v0.1_768p": Row(6.0, 3.0, frozenset({4})),
    # v1.1 of the 768p student, adopted 2026-08-23 when v1.0 left the disk.
    # INHERITED from the v1.0 row by filename family, not attested: see
    # UNATTESTED below, which is what keeps that fact from going quiet.
    "turbo_4step_v1.1_768p": Row(6.0, 3.0, frozenset({4})),
}

#: LEGAL rows no vendor source states, and why. `parse_vendor_table` looks each
#: LEGAL key up in the README and SKIPS the misses, so without this a row with
#: no vendor backing is not caught being wrong -- it is simply never graded,
#: which reads exactly like a row that passed. Declaring it converts a silent
#: hole into a visible one.
#:
#: Being in here is not permission. `vendor_table_agrees` fails if a key is
#: neither found in a vendor source NOR declared here, and fails again if a
#: declared key IS found -- a vendor who publishes the row makes the
#: declaration stale, and a stale one hides a source that now exists.
UNATTESTED = {
    "turbo_4step_v1.1_768p":
        "lightx2v published this file on 2026-08-20 with no README row and it "
        "still has none (checked 2026-08-23). Its 6/3 shift and 4 steps are "
        "inherited from the 4-step v1.0 768p row on the strength of the "
        "filename family, not attested by any vendor source",
}

#: Arms this repo renders at settings the vendor does not recommend, keyed by
#: LEGAL row. The vendor row in `LEGAL` is NOT rewritten to match -- it stays
#: the distilled truth, gradeable against the vendor -- and this records what
#: we actually run beside it.
#:
#: All four fields move together on purpose. Filename, shift, steps and
#: strength are one configuration: a graph at the right steps and the wrong
#: strength is as much a different arm as one at the wrong shift, and grading
#: them separately is how three of the four stay right while the fourth drifts.
#: `shift` is deliberately the DISTILLED value here -- the recipe moves steps
#: and strength, never the schedule the student was fitted to.
OWNER_RECIPE = {
    "turbo_4step_v1.1_768p": {
        "steps": 6,
        "strength": 0.75,
        "shift": (6.0, 3.0),
        "why": "owner's own trials, 2026-08-23, provisional and unscored -- "
               "six steps at 0.75 preferred over the vendor's 4 NFE at 1.0. "
               "Six does NOT divide the 1,000-step grid, so these graphs are "
               "not exact vendor-grid arms; check_distill_grid.py routes them "
               "down its owner-recipe path rather than loosening the tolerance",
    },
}

# LightX2V's inference configs, each of which names the LoRA it loads and
# the shifts it runs at. The second vendor source, for rows the README lacks.
LIGHTX2V_CONFIGS = REPO / "coderef" / "LightX2V" / "configs" / "minimax_h3" / "dmd"

# The base checkpoint's own training shifts. Every graph that loads no turbo
# LoRA must sit here, which is what makes "every shipped graph" true rather
# than "the two graphs that happen to load a LoRA".
BASE_SHIFT = (12.0, 3.0)

#: The vendor's own default for every turbo arm without an OWNER_RECIPE.
#: `--lora-scale` defaults to 1.0 in Minimax-H3-Turbo's inference script.
DEFAULT_TURBO_STRENGTH = 1.0

# turbo_<n>step_<version>[_<resolution>] -- parsed structurally, NOT by
# substring. A substring match classifies a hypothetical
# `turbo_8step_v1.0_768p` as the 12/3 `turbo_8step_v1.0` row, because the
# latter is a prefix of the former. That is a 768p checkpoint silently
# inheriting the wrong shift, i.e. exactly the failure this file exists to
# catch, committed by the file itself.
_NAME = re.compile(r"turbo_(\d+)step_(v\d+\.\d+)(?:_(\d+p))?", re.IGNORECASE)


# A third-party family from ComfyUI-MiniMax-H3-Turbo, named
# `turbo_v<major>_step<checkpoint>_ema` -- version before step, and the number
# after `step` is a training checkpoint, NOT a sampling step count. Reading it
# as one would grade an 8-step graph against "600 steps" and pass anything.
_PACK_NAME = re.compile(r"turbo_v(\d+)_step(\d+)(?:_ema)?", re.IGNORECASE)

# Its README, not our preference: 4 is the minimum, 4-8 the useful range, past
# 8 it stops helping and over-sharpens. Shift is NOT changed -- the pack's
# generate.py hardcodes 12/3 and its example graph carries no shift node.
PACK_STEPS = (4, 8)
PACK_SHIFT = (12.0, 3.0)


#: The two places a generated note claims a version FOR THE GRAPH IT IS ON.
#: Notes legitimately name other LoRAs -- the comparison table lists all five --
#: so a blanket "no note may mention another version" would be red on correct
#: state. These two phrasings are the ones that are about *this* graph, and
#: they are the ones that go stale when `TURBO_768P_LORA` moves.
_THIS_GRAPH_CLAIMS = (
    re.compile(r"This graph loads the \*\*([^*]+?)\*\* LoRA"),
    re.compile(r"\|\s*([^|]+?)\s*\(this graph\)\s*\|"),
)


def note_versions(doc) -> list[str]:
    """Every version label a UI graph's notes claim for itself."""
    out = []
    for node in doc.get("nodes", []):
        if node.get("type") != "MarkdownNote":
            continue
        text = " ".join(str(w) for w in (node.get("widgets_values") or []))
        for pattern in _THIS_GRAPH_CLAIMS:
            out.extend(m.strip() for m in pattern.findall(text))
    return out


# Parallel Decoding Distillation, converted by bench/convert_pdd_lora.py.
# Graded differently from every turbo row and deliberately so: PDD carries no
# shift of its own to inherit, because its block boundaries ARE the base
# checkpoint's own schedule -- so the shift must be the BASE one, and the step
# count must be the `nfe` the converted artifact was actually fused for.
#
# That step count is read from the FILE, not from h3_config. A PDD arm at the
# wrong step count evaluates the model off the boundaries its fused heads were
# built for, and the only other thing that would notice is a runtime warning
# in a log nobody reads. Grading it against our own constant would be the
# check deriving its expectation from the thing it is checking.
_PDD_NAME = re.compile(r"_pdd_(\d+)step_", re.IGNORECASE)


def classify_pdd(lora_name):
    return bool(_PDD_NAME.search(lora_name.replace("-", "_")))


def lora_path(name):
    """Absolute path for a LoRA a graph names, or None.

    Prefers ComfyUI's own resolver so `extra_model_paths.yaml` is honoured,
    and falls back to the stock directory. Same shape as
    `bench/check_lora_alpha.py`'s resolver; kept local rather than imported
    because this file otherwise needs no ComfyUI at all and a shared helper
    would drag that requirement into every case here.
    """
    comfy_root = HERE.parents[2]
    try:
        sys.path.insert(0, str(comfy_root))
        import folder_paths  # noqa: WPS433
        found = folder_paths.get_full_path("loras", name)
        if found:
            return Path(found)
    except Exception:
        pass
    candidate = comfy_root / "models" / "loras" / name
    return candidate if candidate.exists() else None


def pdd_grid(lora_name):
    """`pdd_num_steps` from the converted file, or None."""
    return _pdd_meta(lora_name, "pdd_num_steps")


def pdd_nfe(lora_name):
    """`pdd_nfe` from the converted file's own metadata, or None if unreadable.

    None is not a pass: the caller fails on it. A PDD arm whose artifact cannot
    be read is one whose schedule cannot be graded, and the filename's own
    `_8step_` is not evidence -- the converter writes both, and only the
    metadata is what the node actually consumes.
    """
    return _pdd_meta(lora_name, "pdd_nfe")


def _pdd_meta(lora_name, key):
    import json as _json
    import struct as _struct
    path = lora_path(lora_name)
    if path is None:
        return None
    try:
        with open(path, "rb") as handle:
            n = _struct.unpack("<Q", handle.read(8))[0]
            meta = _json.loads(handle.read(n)).get("__metadata__") or {}
        return int(meta[key])
    except Exception:
        return None


def classify_pack(lora_name):
    """The pack family, or None. Deliberately separate from `classify`: these
    are graded against a step RANGE and a fixed shift, not a single row."""
    return bool(_PACK_NAME.search(lora_name.replace("-", "_")))


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

def _literal(doc, value):
    """The value behind an API-form input, or None if it cannot be read.

    An input is either a literal or a `[node_id, slot]` link, and this
    returned None for every link until it was pointed at the resolver --
    coercing one with `float()` raises TypeError, which surfaces as a FAIL on a
    graph that is actually fine. That traded one wrong answer for a quieter
    one: the day anyone wires
    a `strength_model` or a `shift_video` from a constant node, every graph
    that does it reads here as ungradeable and this file goes red on correct
    state.

    So the link is followed, once, by `h3_config.resolve_link`, which is the
    only walker in the repo. None still means "could not read", and it now
    means what it says: the chain is genuinely unresolvable -- a computed
    output, a class the resolver has no row for, or a broken link. The caller's
    existing "could not read" path is unchanged for all three, deliberately;
    telling them apart is `GraphValue.reason`'s job and no case here needs it
    yet.
    """
    got = resolve_link(doc, value)
    return got.value if got.ok else None


def read_api(doc) -> Found:
    """API form: {node_id: {class_type, inputs{}}}."""
    loras: list[str] = []
    shift: tuple[float, float] | None = None
    steps: int | None = None
    scheduler: str | None = None
    shifts: list[tuple[float, float]] = []
    strengths: dict[str, float] = {}
    pdd_nfe: int | None = None
    for node in doc.values():
        ct, inp = node.get("class_type"), node.get("inputs", {})
        if ct in LORA_LOADER_CLASSES:
            # The pack node is a turbo loader too. Matching only the stock
            # one made its graphs read as BASE graphs -- policed for shift,
            # which they happened to satisfy, and never graded on steps.
            # A pass for the wrong reason is what this file exists to stop.
            loras.append(inp.get("lora_name", ""))
            s = _literal(doc, inp.get("strength_model"))
            if s is None:
                s = _literal(doc, inp.get("strength"))     # MiniMaxH3PDDLoRA
            if ct == "MiniMaxH3PDDLoRA":
                pdd_nfe = _literal(doc, inp.get("nfe")) or None
            if s is not None:
                strengths[str(inp.get("lora_name", ""))] = float(s)
        elif ct == "MiniMaxH3SigmaShift":
            sv, sa = (_literal(doc, inp.get("shift_video")),
                      _literal(doc, inp.get("shift_audio")))
            shift = None if sv is None or sa is None else (float(sv), float(sa))
            if shift is not None:
                shifts.append(shift)
    # Steps and scheduler come from `h3_config.graph_schedule`, not from a
    # local `BasicScheduler` branch: since 0.83.0 a PDD graph carries no
    # scheduler node at all -- `MiniMaxH3PDDLoRA` emits SIGMAS -- and reading
    # only `BasicScheduler` reported "could not read the sampler's step count"
    # on every PDD graph, which this file treats as a failure.
    steps, scheduler = graph_schedule(doc)
    return Found(loras, shift, steps, scheduler, tuple(shifts), strengths, pdd_nfe)


def read_ui(doc) -> Found:
    """UI form: widgets_values positionally. LoraLoaderModelOnly is
    [name, strength]; MiniMaxH3SigmaShift is [video, audio]; BasicScheduler
    is [scheduler, steps, denoise]."""
    loras: list[str] = []
    shift: tuple[float, float] | None = None
    steps: int | None = None
    scheduler: str | None = None
    shifts: list[tuple[float, float]] = []
    strengths: dict[str, float] = {}
    pdd_nfe: int | None = None
    for node in doc.get("nodes", []):
        t, w = node.get("type"), node.get("widgets_values") or []
        if t in LORA_LOADER_CLASSES and w:
            loras.append(w[0])
            if len(w) >= 2 and isinstance(w[1], (int, float)):
                strengths[str(w[0])] = float(w[1])
            # MiniMaxH3PDDLoRA widgets:
            #   [name, strength, patch_heads, nfe, steps]
            # `steps` was appended 2026-08-28. Kept accurate because the
            # next person inserting a widget reads THIS list, and
            # check_pdd_sigmas::case_ui_and_api_agree exists to catch
            # exactly the mis-index that follows from trusting a stale one.
            if t == "MiniMaxH3PDDLoRA" and len(w) >= 4 and isinstance(w[3], int):
                pdd_nfe = w[3] or None
        elif t == "MiniMaxH3SigmaShift" and len(w) >= 2:
            shift = (float(w[0]), float(w[1]))
            shifts.append(shift)
    # See the matching note in `read_api`.
    steps, scheduler = graph_schedule(doc)
    return Found(loras, shift, steps, scheduler, tuple(shifts), strengths, pdd_nfe)


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


def parse_lightx2v_configs(config_dir: Path):
    """(shift_video, shift_audio, steps) per LEGAL key, from LightX2V's configs.

    Each config's `lora_configs[].path` names the LoRA file, which `classify`
    keys the same way it keys a graph's. Shifts are `video_flow_shift` /
    `audio_flow_shift`. Steps are `infer_steps - 1`: LightX2V runs an N-step
    DMD LoRA at N+1 evaluations (`h3_step_update: training_euler`), and the
    caller verifies that convention against the README rather than trusting
    this docstring.
    """
    rows: dict[str, set] = {}
    for path in sorted(config_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for lora in doc.get("lora_configs", []):
            key = classify(lora.get("path", ""))
            if key is None:
                continue
            sv, sa = float(doc["video_flow_shift"]), float(doc["audio_flow_shift"])
            rows.setdefault(key, set()).add((sv, sa, int(doc["infer_steps"]) - 1))
    out = {}
    for key, found in rows.items():
        assert len(found) == 1, (
            f"LightX2V's configs disagree with each other about {key}: {sorted(found)}")
        sv, sa, steps = next(iter(found))
        out[key] = (sv, sa, frozenset({steps}))
    return out


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

        # The second source. Its step convention is graded on every row the
        # two sources share before any README-less row is believed: a config
        # whose `infer_steps - 1` did not reproduce the README's count would
        # mean the convention is wrong, and the SLA row would inherit a
        # wrong step count from a docstring.
        cfg_rows = parse_lightx2v_configs(LIGHTX2V_CONFIGS)
        shared = set(rows) & set(cfg_rows)
        assert shared, (
            "no LoRA appears in both the Turbo README and LightX2V's configs; "
            "the N+1 step convention cannot be verified, so the config-only "
            "rows cannot be graded")
        for key in shared:
            sv, sa, steps = cfg_rows[key]
            assert (sv, sa) == rows[key][:2], (
                f"{key}: LightX2V config shift {sv}/{sa} disagrees with the "
                f"README's {rows[key][0]}/{rows[key][1]}")
            assert steps <= rows[key][2], (
                f"{key}: LightX2V runs {sorted(steps)[0] + 1} evaluations, "
                f"which is not one more than any README count "
                f"{sorted(rows[key][2])}; the N+1 convention does not hold")
        for key in set(cfg_rows) - set(rows):
            sv, sa, steps = cfg_rows[key]
            want = LEGAL[key]
            assert (sv, sa) == (want.shift_video, want.shift_audio), (
                f"{key}: LightX2V config says shift {sv}/{sa}, our table says "
                f"{want.shift_video}/{want.shift_audio}")
            assert steps == want.steps, (
                f"{key}: LightX2V config implies steps {sorted(steps)}, our "
                f"table says {sorted(want.steps)}")
        missing = set(LEGAL) - set(rows) - set(cfg_rows)
        undeclared = sorted(missing - set(UNATTESTED))
        assert not undeclared, (
            f"neither the vendor README nor LightX2V's configs have a row for "
            f"{undeclared}. Either the source moved, or this row is inherited "
            f"rather than attested -- in which case declare it in UNATTESTED "
            f"with the reason, so the gap is visible instead of silent.")

        # The other direction. A declared row that a vendor source DOES carry
        # means the declaration outlived the gap it described, and a stale one
        # hides a source that now exists -- so the row would keep being graded
        # as "inherited" while the real numbers sat unread a directory away.
        stale = sorted(set(UNATTESTED) & (set(rows) | set(cfg_rows)))
        assert not stale, (
            f"{stale} is declared UNATTESTED but a vendor source now carries "
            f"it. Drop the declaration so the row is graded against the "
            f"source rather than against its filename family.")

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

        # Every (lora, steps, shift) triple the config defines, not just the
        # first. A second turbo LoRA added as constants and never graded is
        # the same silent-shift bug one indirection further out.
        triples = [("TURBO", h3_config.TURBO_LORA, h3_config.TURBO_STEPS,
                    h3_config.TURBO_SHIFT)]
        if hasattr(h3_config, "TURBO_768P_LORA"):
            triples.append(("TURBO_768P", h3_config.TURBO_768P_LORA,
                            h3_config.TURBO_768P_STEPS, h3_config.TURBO_768P_SHIFT))
        if hasattr(h3_config, "TURBO_REF2VA_LORA"):
            triples.append(("TURBO_REF2VA", h3_config.TURBO_REF2VA_LORA,
                            h3_config.TURBO_REF2VA_STEPS,
                            h3_config.TURBO_REF2VA_SHIFT))
        if hasattr(h3_config, "TURBO_SLA_LORA"):
            triples.append(("TURBO_SLA", h3_config.TURBO_SLA_LORA,
                            h3_config.TURBO_SLA_STEPS, h3_config.TURBO_SLA_SHIFT))

        # Catch a triple being added to the config without being added here.
        declared = {n for n in dir(h3_config)
                    if n.endswith("_LORA") and "TURBO" in n}
        # TURBO_PACK_LORA is graded, but by classify_pack against a step range
        # rather than by a vendor row, so it is asserted here directly instead
        # of joining `triples`.
        if hasattr(h3_config, "TURBO_PACK_LORA"):
            assert classify_pack(h3_config.TURBO_PACK_LORA), (
                f"TURBO_PACK_LORA {h3_config.TURBO_PACK_LORA!r} does not parse "
                "as the pack's turbo_v<n>_step<ckpt> family")
            lo, hi = PACK_STEPS
            assert lo <= h3_config.TURBO_PACK_STEPS <= hi, (
                f"TURBO_PACK_STEPS {h3_config.TURBO_PACK_STEPS} outside the "
                f"documented {lo}-{hi}")
            assert h3_config.TURBO_PACK_SCHEDULER == "simple", (
                "the pack documents `simple` and nothing else")
            assert h3_config.TURBO_PACK_STRENGTH == 1.0, (
                "the pack tunes for strength 1.0 across its whole step range")

        # NOT intersected with `declared`. It used to be
        # `{...} & declared`, which made `graded` a subset of `declared` by
        # construction, so the assert below could only ever catch a constant
        # being ADDED. A rename or a removal dropped the name from both sides
        # at once and the check stayed green while silently grading one triple
        # fewer -- coverage narrowing with no signal, which is worse than a
        # red. Comparing the literal set catches both directions.
        graded = {"TURBO_LORA", "TURBO_768P_LORA", "TURBO_SLA_LORA",
                  "TURBO_PACK_LORA", "TURBO_REF2VA_LORA"}
        assert declared == graded, (
            f"turbo LoRA constants and this check disagree. Declared in "
            f"h3_config but not graded here: {sorted(declared - graded)}. "
            f"Graded here but no longer in h3_config: {sorted(graded - declared)} "
            f"-- if one was renamed, rename it here too rather than deleting "
            f"the row, or this check quietly stops covering it.")

        for label, lora, steps, shift in triples:
            key = classify(lora)
            assert key, f"{label}_LORA {lora!r} matches no known row"
            want = LEGAL[key]
            got = (float(shift["shift_video"]), float(shift["shift_audio"]))
            assert got == (want.shift_video, want.shift_audio), (
                f"{label} ({key}): config shift {got}, distilled at "
                f"({want.shift_video}, {want.shift_audio})")
            recipe = OWNER_RECIPE.get(key)
            if recipe is None:
                assert steps in want.steps, (
                    f"{label} ({key}): config steps {steps}, "
                    f"distilled for {sorted(want.steps)}")
            else:
                # An owner recipe replaces the STEP claim, never the shift: the
                # shift assertion above already ran and is the distilled value.
                assert steps == recipe["steps"], (
                    f"{label} ({key}): config steps {steps}, but OWNER_RECIPE "
                    f"declares {recipe['steps']}. Move both or neither -- a "
                    f"recipe that disagrees with the config it describes is "
                    f"worse than no recipe.")
                assert got == recipe["shift"], (
                    f"{label} ({key}): OWNER_RECIPE shift {recipe['shift']} is "
                    f"not the config's {got}")
                assert recipe["shift"] == (want.shift_video, want.shift_audio), (
                    f"{label} ({key}): OWNER_RECIPE moved the SHIFT off the "
                    f"distilled {(want.shift_video, want.shift_audio)}. Steps "
                    f"and strength are the recipe; the schedule the student "
                    f"was fitted to is not.")

    check("h3_config turbo triples are legal", config_is_consistent)

    # ---- every shipped graph, both forms ---------------------------------
    def graphs_are_consistent():
        turbo_graphs, base_graphs = {}, {}
        for path in graph_paths(WORKFLOWS):
            doc = json.loads(path.read_text(encoding="utf-8"))
            found = read_api(doc) if "nodes" not in doc else read_ui(doc)
            turbo = [l for l in found.loras if is_turbo(l) or classify_pdd(l)]

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
                # PDD: base shift, and the step count the artifact was fused
                # for. Not a turbo row and not the pack range.
                if classify_pdd(lora):
                    assert found.shift == BASE_SHIFT, (
                        f"{path.name}: {lora} is a PDD arm, whose block "
                        f"boundaries ARE the base schedule -- it must sit at "
                        f"{BASE_SHIFT[0]}/{BASE_SHIFT[1]}, has {found.shift}")
                    # **The SAMPLER's step count is the evaluation count.**
                    # Since 2026-08-27 the node reads the block boundaries off
                    # `sample_sigmas` at run time, so the graph's `steps` is
                    # what runs and the file's `pdd_nfe` is only a fallback for
                    # a sampler that publishes no schedule. This used to grade
                    # `steps` against the file and went red on every correct
                    # 4-step arm the moment the widget stopped carrying 4.
                    grid = pdd_grid(lora)
                    assert grid, (
                        f"{path.name}: could not read `pdd_num_steps` from "
                        f"{lora}. A PDD arm whose grid cannot be read cannot "
                        "be graded; the filename is not evidence.")
                    assert found.steps is not None, (
                        f"{path.name}: could not read the sampler's step count, "
                        f"which IS the evaluation count since 2026-08-27. A PDD "
                        f"arm whose schedule cannot be read cannot be graded; "
                        f"the filename is not evidence.")
                    assert grid % found.steps == 0, (
                        f"{path.name}: {found.steps} evaluations do not divide "
                        f"the file's {grid}-point grid, so the blocks come out "
                        f"uneven. The node takes them anyway and says so, but a "
                        f"SHIPPED arm should tile: "
                        f"{sorted(n for n in range(1, grid + 1) if grid % n == 0)}.")
                    # `nfe` is an override that forces uniform blocks and
                    # ignores the schedule. Legal, but it means the arm decodes
                    # one partition while stepping another -- an experiment, not
                    # something to ship. Every shipped graph carries 0.
                    assert not found.pdd_nfe or found.pdd_nfe == found.steps, (
                        f"{path.name}: the node's `nfe` override is "
                        f"{found.pdd_nfe} while the sampler runs {found.steps} "
                        f"steps. That decodes the blocks of one partition while "
                        f"stepping another. Leave `nfe` at 0 so it follows the "
                        f"schedule, or change the sampler's steps to match.")
                    continue
                # The third-party family is graded against a step RANGE and
                # the unchanged base shift, not against a single vendor row.
                if classify_pack(lora):
                    assert found.shift == PACK_SHIFT, (
                        f"{path.name}: {lora} does not carry its own shift -- "
                        f"its generate.py hardcodes {PACK_SHIFT[0]}/{PACK_SHIFT[1]} "
                        f"and its example graph has no shift node at all -- "
                        f"but the graph sits at {found.shift}")
                    assert found.steps is not None, (
                        f"{path.name}: loads {lora} but no BasicScheduler step "
                        "count could be read")
                    lo, hi = PACK_STEPS
                    assert lo <= found.steps <= hi, (
                        f"{path.name}: {lora} is documented for {lo}-{hi} steps "
                        f"({lo} the minimum, past {hi} it over-sharpens); "
                        f"graph has {found.steps}")
                    continue
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
                # Steps and strength come from the owner recipe where one is
                # declared, and from the vendor row otherwise. The SHIFT above
                # is graded against the vendor either way -- a recipe never
                # moves the schedule the student was fitted to.
                recipe = OWNER_RECIPE.get(key)
                if recipe is None:
                    assert found.steps in want.steps, (
                        f"{path.name}: {key} wants steps {sorted(want.steps)}, "
                        f"graph has {found.steps}")
                else:
                    assert found.steps == recipe["steps"], (
                        f"{path.name}: {key} runs the owner recipe at "
                        f"{recipe['steps']} steps ({recipe['why']}), graph has "
                        f"{found.steps}")
                # Strength, graded for every turbo arm and not only the recipe
                # ones. Read by nothing until 2026-08-23.
                got_strength = (found.strengths or {}).get(lora)
                assert got_strength is not None, (
                    f"{path.name}: loads {key} but no strength_model could be "
                    f"read; filename, shift, steps and strength are one "
                    f"configuration and three of four is not a pass")
                want_strength = (recipe["strength"] if recipe
                                 else DEFAULT_TURBO_STRENGTH)
                assert got_strength == want_strength, (
                    f"{path.name}: {key} wants strength {want_strength:g}, "
                    f"graph has {got_strength:g}"
                    + (f" ({recipe['why']})" if recipe else ""))

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
        # The SLA file: `_768p` must land it on its own 6/3 row and not on
        # the 544p v0.1 row whose name is a prefix of it. Same trap as above,
        # now with a real file on disk on each side of it.
        assert classify("h3/lightx2v_Minimax-h3-Turbo-SLA/"
                        "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors") \
            == "turbo_4step_v0.1_768p", "the SLA file must resolve to its own 768p row"
        assert classify("h3/lightx2v_Minimax-h3-Turbo/"
                        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors") \
            == "turbo_4step_v0.1", "the 544p v0.1 family must not be pulled onto the 768p row"
        # v1.1 768p classified as None until 2026-08-23, on the rule that a
        # file with no vendor row must not inherit v1.0's shift. The owner
        # adopted it anyway when v1.0 left this disk, so it now HAS a row --
        # and the thing that assertion was protecting moved to `UNATTESTED`,
        # which states the inheritance instead of forbidding it and fails if
        # the vendor ever publishes the real numbers.
        #
        # Kept as a positive assertion rather than deleted: the reason to write
        # it down is that classification is what hands a file its shift, and
        # this is the one row where that shift came from a filename.
        assert classify("h3/lightx2v_Minimax-h3-Turbo/"
                        "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors") \
            == "turbo_4step_v1.1_768p", (
            "v1.1 768p must resolve to its OWN row, never to v1.0's -- the "
            "shift is the same by inheritance, and a row of its own is what "
            "keeps UNATTESTED able to say so")
        assert "turbo_4step_v1.1_768p" in UNATTESTED, (
            "v1.1's row is inherited, not attested. If a vendor source now "
            "carries it, drop the UNATTESTED entry -- do not silently keep "
            "grading it against its filename")

    # ---- the note against the file it sits beside -------------------------
    def notes_match_the_lora():
        """A graph cannot load one version while its own note names another.

        The generated help text and the `lora_name` widget were independent
        strings until 2026-08-23. `TURBO_768P_LORA` moved to v1.1 and sixteen
        graphs went on loading v1.1 under notes that still read v1.0 -- correct
        file, wrong instructions, and nothing in the suite looked at the note
        at all. `h3_config.turbo_label()` now derives the displayed label from
        the same filename the graph loads; this is what keeps them derived.

        Only claims a note makes about ITS OWN graph are graded. The comparison
        table naming the other four LoRAs is correct and must stay green.
        """
        graded = skipped_pack = 0
        problems = []
        for path in graph_paths(WORKFLOWS):
            doc = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(doc.get("nodes"), list):
                continue  # the API form carries no notes
            found = read_ui(doc)
            turbo = [n for n in found.loras if is_turbo(n)]
            if not turbo:
                continue
            want = turbo_label(turbo[0])
            if not want:
                # The third-party pack's `turbo_v4_step600_ema` family has no
                # N-step-vX.Y label. Counted, not silently passed.
                skipped_pack += 1
                continue
            claimed = note_versions(doc)
            if not claimed:
                continue
            graded += 1
            wrong = sorted({c for c in claimed if c != want})
            if wrong:
                problems.append(
                    f"{path.relative_to(REPO)} loads {want!r} but its note "
                    f"claims {wrong} for this graph")
        assert graded, (
            "no graph paired a turbo LoRA with a self-describing note; this "
            "case would pass on an empty set, so it has lost its subject")
        assert not problems, "; ".join(problems)
        print(f"        ({graded} graph(s) with a self-describing note, "
              f"{skipped_pack} pack graph(s) with no parseable label)")

    check("notes match the lora they sit beside", notes_match_the_lora)

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
