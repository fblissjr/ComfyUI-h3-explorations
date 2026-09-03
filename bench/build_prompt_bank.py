#!/usr/bin/env python
"""Grade every prompt in `prompt_bank/` and derive `docs/prompt_bank.md`.

## What the bank is

`prompt_bank/*.txt` are HOUSE-authored H3 prompts, one per file, each written
to exercise a named part of the guides' structure: a mode, a frame count, a
camera motion, a speech shape, a marker, a reference task type. They are
conformant drafts to start from and probe prompts for the audit list; nothing
about them is attested. **An entry the generator ships (the `ships` column) has
rendered as its graphs have, since 2026-09-03; the rest have not**, and a mechanical pass says
nothing about whether register, density or pacing land the way the vendor's
own five worked examples do. The manifest `prompt_bank/bank.json` names each
prompt's mode, frame count and, for ref2va, the shipped donor graph whose
sockets its labels are graded against.

## Why a generator, and why `--check` is a gate

The first bank lived in a gitignored file whose hand-written coverage table
was wrong on the day it was written (it said fifteen frame counts were used;
its own rows listed fourteen), and its prompts could go stale against the
grader silently -- the grader changed twice that day. No existing gate reads
loose prompts. So this does three things, and `--check` fails on any of them:

1. every prompt grades 0 FAIL and 0 WARN through `grade_prompt_text.grade_text`
   at its declared frame count and donor -- the SAME function the CLI runs, so
   the bank is graded by exactly what an author grades by;
2. the manifest and the directory agree, every frame count is on the grid,
   every ref2va entry names a donor, and every mode is present;
3. `docs/prompt_bank.md` is current.

The coverage tables in that file are DERIVED from the prompt text against the
guides' closed sets, never typed. Camera motions are recognised from the prose
forms this bank writes them in (`pushes in`, `holds a static shot`, ...); that
mapper is a house convention for reading THIS corpus, not the vocabulary gate,
which is `bench/check_camera_vocabulary.py`.

Adding a prompt: write `prompt_bank/<id>.txt`, add its manifest entry, run
this script, commit all three.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workflows"))
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO))

import h3_rules  # noqa: E402
import preflight_graph as pf  # noqa: E402
import grade_prompt_text as gpt  # noqa: E402

BANK = REPO / "prompt_bank"
MANIFEST = BANK / "bank.json"
OUT = REPO / "docs" / "prompt_bank.md"
MODES = ("t2va", "i2va", "fl2va", "l2va", "ref2va")

# base_en 4.3's twenty motion names, keyed by the prose form this bank writes.
MOTION_PROSE = OrderedDict([
    ("Zoom In", r"\bzooms? in\b"), ("Zoom Out", r"\bzooms? out\b"),
    ("Push In", r"\bpush(?:es)? in\b"), ("Pull Out", r"\bpulls? out\b"),
    ("Pan Left", r"\bpans? left\b"), ("Pan Right", r"\bpans? right\b"),
    ("Truck Left", r"\btrucks? left\b"), ("Truck Right", r"\btrucks? right\b"),
    ("Tilt Up", r"\btilts? up\b"), ("Tilt Down", r"\btilts? down\b"),
    ("Pedestal Up", r"\bpedestals? up\b"), ("Pedestal Down", r"\bpedestals? down\b"),
    ("Arc Shot", r"\barc shot\b"), ("Tracking Shot", r"\btracking shot\b"),
    ("Static Shot", r"\bstatic shot\b"),
    ("Shake Slightly", r"\bshak(?:es|ing) slightly\b"),
    ("Shake Strongly", r"\bshak(?:es|ing) strongly\b"),
    ("POV", r"\bPOV\b|\bpoint of view\b"),
    ("Roll Clockwise", r"\brolls? clockwise\b"),
    ("Roll Counterclockwise", r"\brolls? counter-?clockwise\b"),
])
MODIFIERS = ("with small amplitude", "with large amplitude",
             "at slow speed", "at fast speed")
CUTS = ("the camera cuts to", "the shot cuts to", "the shot transitions to",
        "the shot changes to", "the shot switches to")
REQUESTED = ("cross-dissolve", "fade", "wipe")
STYLES = ("Cinematic", "live-action", "2D-animated", "3D CG", "claymation",
          "watercolor", "vintage film")
TASK_TYPES = ("keyframe completion", "reference generation", "video editing",
              "video continuation", "audio reuse", "audio reference")
LEGAL_FRAMES = [h3_rules.FRAME_FACTOR * k + h3_rules.FRAME_REMAINDER
                for k in range(5, 22)]
assert LEGAL_FRAMES[0] == 90 and LEGAL_FRAMES[-1] == h3_rules.MAX_LENGTH
MAIN = {"ref2va": "detailed_description"}
H3_MARKERS = ("<d>",) + pf.ALL_MARKERS


def field(text: str, name: str) -> str:
    """Content of a section: same line as the label, or the lines after it."""
    m = re.search(rf"^{name}:[ \t]*(.*)$", text, re.M)
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(r"^[a-z_]+:", rest, re.M)
    body = m.group(1) + "\n" + (rest[:nxt.start()] if nxt else rest)
    return body.strip()


def facts(text: str, mode: str) -> dict:
    main = field(text, MAIN.get(mode, "integrated_multimodal_description"))
    ids = re.findall(r"\((S\d+(?:,S\d+)*)\)", main)
    speakers = sorted({s for grp in ids for s in grp.split(",")},
                      key=lambda s: int(s[1:]))
    if mode == "ref2va":
        opening = main[:main.find("[Shot 1]")] if "[Shot 1]" in main else main[:300]
    else:
        i = main.find("[Shot 1]")
        opening = main[i:i + 220] if i >= 0 else main[:220]
    music = field(text, "non_diegetic_music")
    sound = field(text, "overall_soundscape")
    summary = field(text, "summary")
    tt = re.match(r"\[([^\]]+)\]", summary)
    retention = field(text, "retention_analysis")
    return {
        "words": len(main.split()),
        "shots": len(re.findall(pf.SHOT_HEADER_RE, main)),
        "speakers": speakers,
        "compound": any("," in g for g in ids),
        "d_blocks": len(re.findall(r"<d>", main)),
        "languages": sorted(set(re.findall(r"<d>\s*\[([A-Z][a-z]+)\]", main))),
        "motions": [n for n, rx in MOTION_PROSE.items() if re.search(rx, main, re.I)],
        "modifiers": [m for m in MODIFIERS if m in main],
        "cuts": [c for c in CUTS if c in main],
        "requested": [r for r in REQUESTED if r in main.lower()],
        "styles": [s for s in STYLES if s.lower() in opening.lower()],
        "on_screen_text": re.findall(r'"([^"\n]{1,80})"', re.sub(r"<d>.*?</d>", "", main, flags=re.S)),
        "markers": [m for m in H3_MARKERS if m in main],
        "voiceover": "says in an off-screen voiceover" in main,
        "off_screen_speaker": bool(re.search(r"off-screen(?!\s+voiceover)", main)),
        "unclear": "[unclear]" in main,
        "audio_as_source": bool(re.search(r"<Audio \d+> reaches the phrase", main)),
        "music": "N/A" if music.strip() == "N/A" else ("scored" if music else "absent"),
        "soundscape_na": sound.strip() == "N/A",
        "task_types": [t.strip() for t in tt.group(1).split("+")] if tt else [],
        "retention_markers": sorted(set(re.findall(
            r":\s*(fully_preserved|partially_preserved|attribute_transfer|weak_reference|fully_copy|partially_copy|reference)\b", retention))),
        "labels": sorted(set(re.findall(r"<(?:Picture|Video|Audio) \d+>", text)),
                         key=lambda s: (s.split()[0], int(s.split()[1][:-1]))),
        "chars": len(text),
    }


def load() -> tuple[list[dict], list[str]]:
    problems = []
    if not MANIFEST.exists():
        return [], [f"{MANIFEST.relative_to(REPO)} is missing"]
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["prompts"]
    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids in the manifest")
    files = {p.stem for p in BANK.glob("*.txt")}
    for e in entries:
        if e["mode"] not in MODES:
            problems.append(f"{e['id']}: mode {e['mode']!r} is not one of {MODES}")
        if e["frames"] not in LEGAL_FRAMES:
            problems.append(f"{e['id']}: {e['frames']} frames is not on the 17k+5 grid")
        if (e["mode"] == "ref2va") != bool(e.get("donor")):
            problems.append(f"{e['id']}: a donor is required for ref2va and only for ref2va")
        if e["id"] not in files:
            problems.append(f"{e['id']}: prompt_bank/{e['id']}.txt does not exist")
    for stray in sorted(files - set(ids)):
        problems.append(f"prompt_bank/{stray}.txt is not in the manifest")
    return entries, problems


def grade_all(entries: list[dict]) -> list[dict]:
    rows = []
    shipped = shipped_by()
    for e in entries:
        path = BANK / f"{e['id']}.txt"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").strip()
        try:
            r = gpt.grade_text(text, e["mode"], e.get("donor"), e["frames"])
        except SystemExit as ex:  # donor/mode mismatch
            r = {"findings": [("FAIL", str(ex))], "expected": None, "length": None}
        fails = [m for lvl, m in r["findings"] if lvl == "FAIL"]
        warns = [m for lvl, m in r["findings"] if lvl == "WARN"]
        rows.append({**e, "text": text, "facts": facts(text, e["mode"]),
                     "fails": fails, "warns": warns,
                     "duration": h3_rules.duration_of(e["frames"]),
                     "ships": sorted(shipped.get(text, ())),
                     "adaptable": adaptable(text)})
    return rows


_TIMED = re.compile(r"\bAt \d\d:\d\d(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s*(?:s|sec|secs|seconds?)\b"
                    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                    r"thirteen|fourteen|fifteen)-second\b", re.I)


def adaptable(text: str) -> bool:
    """Whether the text can be rendered at another frame count without a
    rewrite: it names no cut time and no duration. Mechanical, so it can be
    wrong in both directions (a prompt can pace itself in prose); it says
    which entries the length knob is free on, not that the result is good."""
    return _TIMED.search(text) is None


def shipped_by() -> dict[str, set[str]]:
    """prompt text -> the graph stems that carry it, from the graphs
    themselves (the catalogue's scanner), so this file never holds a second
    copy of which prompt ships where."""
    import build_prompt_catalogue as cat
    return {k.strip(): v for k, v in cat.scan_graphs().items()}


def shape_problems(rows: list[dict]) -> list[str]:
    out = []
    if not rows:
        return ["the bank is empty -- a green here would be a run on nothing"]
    present = {r["mode"] for r in rows}
    for m in MODES:
        if m not in present:
            out.append(f"no prompt in mode {m}")
    for r in rows:
        f = r["facts"]
        if not f["motions"]:
            out.append(f"{r['id']}: no camera motion recognised -- either the prompt "
                       f"names none or MOTION_PROSE cannot read its phrasing")
        if not f["styles"] and r["mode"] != "ref2va":
            pass  # off-list styles are legal; recorded, not failed
        if f["music"] == "absent":
            out.append(f"{r['id']}: non_diegetic_music is missing")
        if r["mode"] == "ref2va" and not f["task_types"]:
            out.append(f"{r['id']}: summary carries no task-type prefix")
    return out


def cmd(r: dict) -> str:
    like = f" --like {r['donor']}" if r.get("donor") else ""
    return (f"python bench/grade_prompt_text.py --mode {r['mode']} "
            f"--length {r['frames']}{like} prompt_bank/{r['id']}.txt")


def coverage_table(w, title, universe, hits, note="", open_set=False):
    w(f"### {title}")
    w("")
    if note:
        w(note)
        w("")
    w("| value | prompts |")
    w("|---|---|")
    for v in universe:
        ids = hits.get(v, [])
        w(f"| `{v}` | {', '.join(f'`{i}`' for i in ids) if ids else '**none**'} |")
    w("")
    if open_set:
        w("An open set: the rows are what the bank uses, so nothing here can be "
          "\"not exercised\".")
    else:
        missing = [v for v in universe if not hits.get(v)]
        w(f"Not exercised: {', '.join(f'`{v}`' for v in missing) if missing else 'nothing -- every value above appears at least once'}.")
    w("")


def render(rows: list[dict]) -> str:
    o = []
    w = o.append
    order = {m: i for i, m in enumerate(MODES)}
    rows = sorted(rows, key=lambda r: (order[r["mode"]], r["frames"], r["id"]))
    w("# The prompt bank")
    w("")
    w("**Generated by `bench/build_prompt_bank.py`. Do not hand-edit -- rerun it.** "
      "The prompts are the files in [`prompt_bank/`](../prompt_bank/) and the "
      "manifest is [`prompt_bank/bank.json`](../prompt_bank/bank.json); every "
      "table below is derived from that text against the guides' closed sets, "
      "and `--check` fails when any prompt stops grading clean or this file is stale.")
    w("")
    w("**What these are, and what they are not.** House-authored H3 prompts, one per "
      "file, each written to exercise a named part of the structure the guides "
      "describe: a mode, a frame count on the grid, a camera motion, a speech shape, "
      "a marker, a reference task type. **Every one grades 0 FAIL and 0 WARN** through "
      "`bench/grade_prompt_text.py` at the frame count and donor the manifest names, "
      "and that is the whole of what is established: they satisfy the guides' "
      "MECHANICAL rules, the ones a script can decide -- except where the manifest "
      "carries `recorded_findings`, a shipped prompt whose failings `prompt_audit.md` "
      "already adjudicates; those are reported in the table and not gated. **An entry with graphs in its "
      "`ships` column is a shipped prompt: since 2026-09-03 the generator loads every "
      "shipped prompt from this bank by id (`workflows/prompts.py`), so those have "
      "rendered as their graphs have; an entry with none has never been rendered.** "
      "Nothing here says they are in-distribution the way the vendor's own five worked "
      "examples are (`vendor_guides/base_en.md` Cases 1-4, `ref_en.md` section 7), and "
      "a downstream consumer that wants attested specimens should take those, not "
      "these. Use these as conformant drafts to start from, as probe prompts for "
      "[`prompt_audit.md`](prompt_audit.md)'s render list, and as the idea set a "
      "sister project grades the same scenes against. Rule layers are "
      "[`prompting.md`](prompting.md)'s; where a HOUSE convention is applied the "
      "manifest brief says so.")
    w("")
    w("**Two house choices every prompt makes, stated so they are arguable.** "
      "`<scenetrans>` is never written: base 4.4 states it for a line crossing a cut, "
      "but it matches no token the release declares, so the continuity is carried by "
      "the guide's own phrases in prose. `<|cutoff|>` is written piped and tight "
      "against `</d>`, the form the release declares. Both are OPEN in "
      "`prompting.md` section 12, neither has been rendered, and the sister engine "
      "makes the opposite call on both; `prompt_audit.md` item 4 is the render that "
      "would inform either.")
    w("")
    w("**Re-grading one:** the command under each prompt, from the repo root with the "
      "ComfyUI venv's python. A prompt is conformant AT A DURATION: `S.SS` and every "
      "`At MM:SS.mmm` resolve against the snapped frame count, so grading at another "
      "`--length` is expected to fail. The `adapt` column is the mechanical exception: "
      "a prompt that names no cut time and no duration can take another length from "
      "the graph alone. **Adding one:** write the file, add a manifest "
      "entry, run the builder, commit all three. **Shipping one:** name it by id in "
      "`workflows/build_workflows.py`; the text never goes anywhere else. **A "
      "COMPOSED prompt is shipped the other way round:** `_ref_prompt()` builds a "
      "ref2va prompt from the role tables and the two keyframe defaults build their "
      "Part One line from the frame count, so those arrive here as output rather "
      "than as an id -- the generator looks the composed text up and refuses to "
      "build until it is a file here, which is why a new reference combination "
      "fails with the id it wants written. **For a "
      "bridge from another repo:** the manifest is the contract -- mode and frames "
      "per file, donor stem for ref2va.")
    w("")
    n_na = sum(1 for r in rows if r["facts"]["music"] == "N/A")
    w("## Every prompt")
    w("")
    w("| id | mode | frames | s | adapt | ships | donor | words | shots | speakers | languages | camera | music | grade |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        f = r["facts"]
        grade = ("clean" if not r["fails"] and not r["warns"]
                 else f"{len(r['fails'])} FAIL, {len(r['warns'])} WARN")
        if r.get("recorded_findings") and grade != "clean":
            grade += " (recorded; see the entry)"
        w(f"| [`{r['id']}`](#{r['id'].replace('_', '-')}) | {r['mode']} | {r['frames']} "
          f"| {r['duration']:.3f} | {'yes' if r['adaptable'] else 'pinned'} | {len(r['ships'])} "
          f"| {('`' + r['donor'] + '`') if r.get('donor') else '—'} "
          f"| {f['words']} | {f['shots']} | {len(f['speakers'])}{' +compound' if f['compound'] else ''} "
          f"| {', '.join(f['languages']) or '—'} | {', '.join(f['motions'])} | {f['music']} | {grade} |")
    w("")
    w(f"Counted over the prompts above: {len(rows)} prompts, "
      f"{n_na} with `non_diegetic_music: N/A`. The vendor's eight filled-in music "
      f"lines across both guides carry two `N/A`; the shipped corpus, per "
      f"`prompt_catalogue.md`, leans far harder to `N/A` than that, and this bank "
      f"was written against that lean rather than to it.")
    w("")
    w("## Coverage, derived")
    w("")
    def by(key: str) -> dict:
        values = {x for r in rows for x in r["facts"][key]}
        return {v: [r["id"] for r in rows if v in r["facts"][key]] for v in values}
    w("### Modes")
    w("")
    w("| mode | prompts |")
    w("|---|---|")
    for m in MODES:
        w(f"| {m} | {sum(1 for r in rows if r['mode'] == m)} |")
    w("")
    fr = {n: [r["id"] for r in rows if r["frames"] == n] for n in LEGAL_FRAMES}
    coverage_table(w, "Frame counts, every legal value on the 17k+5 grid",
                   LEGAL_FRAMES, fr,
                   "Duration is frames / 24. The Part One line of a keyframe prompt "
                   "carries it to two decimals.")
    mo = by("motions")
    coverage_table(w, "Camera motion types, base 4.3's twenty", list(MOTION_PROSE), mo,
                   "Recognised from the prose forms this bank writes (`pushes in`, "
                   "`holds a static shot`); the vocabulary gate is "
                   "`bench/check_camera_vocabulary.py`.")
    md = by("modifiers")
    coverage_table(w, "Amplitude and speed, base 4.3's closed sets", MODIFIERS, md,
                   "Medium amplitude and normal speed are written by omitting the phrase.")
    cu = by("cuts")
    coverage_table(w, "Cut phrasings, base 4.2's five", CUTS, cu)
    rq = by("requested")
    coverage_table(w, "Transitions permitted only on explicit request, base 4.2", REQUESTED, rq,
                   "Each of these is used only where the manifest brief records that the "
                   "user asked for it.")
    st = by("styles")
    coverage_table(w, "Styles, base 4.1's seven named ones", STYLES, st,
                   "The guide says \"common styles include\", so the list is open; "
                   "prompts using a style off it are legal and are named in their briefs.")
    la = by("languages")
    coverage_table(w, "Language tags used inside `<d>`", sorted(la), la,
                   "Neither guide lists supported languages; the tag mechanism is "
                   "GUIDE-stated and the set of languages here is a house choice.",
                   open_set=True)
    tt = by("task_types")
    coverage_table(w, "ref2va task types, ref 3's six", TASK_TYPES, tt,
                   "`keyframe completion` is claimable only on a graph that wires a "
                   "keyframe node beside references, and no shipped ref2va graph does, "
                   "so the bank cannot carry it.")
    rm = by("retention_markers")
    coverage_table(w, "Retention markers, ref 4.1's four visual and 4.2's four audio",
                   ("fully_preserved", "partially_preserved", "attribute_transfer",
                    "weak_reference", "fully_copy", "partially_copy", "reference"), rm,
                   "`weak_reference` belongs to both sets.")
    mk = by("markers")
    coverage_table(w, "Declared H3 markers present", H3_MARKERS, mk,
                   "`<d>` is the guides'; the other five are house patterns under test "
                   "(`prompting.md` section 7).")
    w("### Speech shapes")
    w("")
    w("| shape | prompts |")
    w("|---|---|")
    shapes = [
        ("no speaker", [r["id"] for r in rows if not r["facts"]["speakers"]]),
        ("one speaker", [r["id"] for r in rows if len(r["facts"]["speakers"]) == 1]),
        ("two speakers", [r["id"] for r in rows if len(r["facts"]["speakers"]) == 2]),
        ("three or more speakers", [r["id"] for r in rows if len(r["facts"]["speakers"]) >= 3]),
        ("a compound id in unison", [r["id"] for r in rows if r["facts"]["compound"]]),
        ("the exact voiceover phrase", [r["id"] for r in rows if r["facts"]["voiceover"]]),
        ("an off-screen speaker who is not a voiceover", [r["id"] for r in rows if r["facts"]["off_screen_speaker"]]),
        ("a `[unclear]` span in reused source words", [r["id"] for r in rows if r["facts"]["unclear"]]),
        ("a lyric cited via `<Audio N>` with no speaker id", [r["id"] for r in rows if r["facts"]["audio_as_source"]]),
        ("on-screen text in double quotes", [r["id"] for r in rows if r["facts"]["on_screen_text"]]),
        ("`overall_soundscape: N/A` (complete silence requested)", [r["id"] for r in rows if r["facts"]["soundscape_na"]]),
    ]
    for name, ids in shapes:
        w(f"| {name} | {', '.join(f'`{i}`' for i in ids) if ids else '**none**'} |")
    w("")
    w("---")
    w("")
    w("## The prompts")
    w("")
    for r in rows:
        f = r["facts"]
        w(f"## {r['id']}")
        w("")
        w(f"**{r['mode']}, {r['frames']} frames, {r['duration']:.3f} s"
          + (f", donor `{r['donor']}`" if r.get("donor") else "") + ".** " + r["brief"]
          + (f" **Recorded findings:** {r['recorded_findings']}" if r.get("recorded_findings") else "")
          + (f" Ships in: " + ", ".join(f"`{g}`" for g in r["ships"]) + "." if r["ships"] else ""))
        w("")
        bits = [f"camera: {', '.join(f['motions'])}"]
        if f["speakers"]:
            bits.append(f"speakers: {', '.join(f['speakers'])}")
        if f["languages"]:
            bits.append(f"languages: {', '.join(f['languages'])}")
        if f["cuts"]:
            bits.append(f"cuts: {', '.join(f['cuts'])}")
        if f["on_screen_text"]:
            bits.append("on-screen text: " + ", ".join(f'"{s}"' for s in f["on_screen_text"]))
        if f["task_types"]:
            bits.append(f"task: [{' + '.join(f['task_types'])}]")
        if f["labels"]:
            bits.append(f"labels: {', '.join(f['labels'])}")
        w("Derived: " + "; ".join(bits) + f"; {f['words']} words in the main field; "
          f"{f['chars']} characters in all.")
        w("")
        w("```")
        w(cmd(r))
        w(f"  {len(r['fails'])} FAIL, {len(r['warns'])} WARN")
        w("```")
        w("")
        w("```text")
        w(r["text"])
        w("```")
        w("")
    return "\n".join(o) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit nonzero on any FAIL/WARN, shape "
                         "problem, or a stale docs/prompt_bank.md")
    args = ap.parse_args()

    entries, problems = load()
    rows = grade_all(entries)
    # A shipped prompt whose findings are already adjudicated (the manifest's
    # `recorded_findings` names where) is reported, never gated: the bank
    # became the only home of shipped text on 2026-09-03, and some shipped
    # prompts do not meet the bank's own bar -- `prompt_audit.md` already
    # says so. Silently passing them would hide that; failing the bank on
    # them would train everyone to ignore red. The table shows the grade.
    recorded = {r["id"] for r in rows if r.get("recorded_findings")}
    shape = shape_problems(rows)
    problems += [m for m in shape if m.split(":")[0] not in recorded]
    # A suppressed SHAPE problem has to be announced too. Until 2026-09-03 the
    # `noted` line below fired only on a FAIL or a WARN, so an entry whose only
    # finding was a shape problem was suppressed in complete silence -- and the
    # composed ref2va arms added that day are exactly that case: four of them
    # grade clean and name no camera motion, because they follow the reference
    # video's camera. A suppression nothing prints is the same defect as a
    # green run over an empty set.
    muted = {}
    for m in shape:
        pid = m.split(":")[0]
        if pid in recorded:
            muted.setdefault(pid, []).append(m)
    red = [(r["id"], m) for r in rows for m in r["fails"] if r["id"] not in recorded]
    amber = [(r["id"], m) for r in rows for m in r["warns"] if r["id"] not in recorded]
    for pid, m in red:
        print(f"  FAIL  {pid}: {m}")
    for pid, m in amber:
        print(f"  WARN  {pid}: {m}")
    for p in problems:
        print(f"  FAIL  shape: {p}")
    for r in rows:
        n_shape = len(muted.get(r["id"], ()))
        if r["id"] in recorded and (r["fails"] or r["warns"] or n_shape):
            print(f"  noted {r['id']}: {len(r['fails'])} FAIL, {len(r['warns'])} WARN, "
                  f"{n_shape} shape, recorded -- {r['recorded_findings']}")
    clean = sum(1 for r in rows if not r["fails"] and not r["warns"])
    print(f"  {clean} of {len(rows)} prompt(s) grade clean; {len(problems)} shape problem(s)")
    if not rows:
        return 2

    out = render(rows)
    if args.check:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        stale = cur != out
        if stale:
            print("  FAIL  docs/prompt_bank.md is stale -- rerun bench/build_prompt_bank.py")
        if red or amber or problems or stale:
            return 1
        print("  ok    every bank prompt grades clean and docs/prompt_bank.md is current")
        return 0

    OUT.write_text(out, encoding="utf-8")
    print(f"  ok    wrote {OUT.relative_to(REPO)}")
    return 1 if (red or amber or problems) else 0


if __name__ == "__main__":
    sys.exit(main())
