#!/usr/bin/env python3
"""Where a masked video stands: one row a shot and a subject, with what the plan wants there, what the delivery
table lays there, and what else has been rendered on those frames.

    <python> bench/shot_coverage.py --shots SHOTS.json --plan PLAN.json --table TABLE.txt [--ledger LEDGER.json] \\
        [--verdicts VERDICTS.jsonl] [--gap-check GAP_CHECK.json] [--json OUT.json]

**What it buys.** On 2026-10-11 nobody could say, per shot and per subject, what the owner had ruled, what
render existed and what was missing: the delivery table was the nearest thing and it is typed by hand, and
the day's renders went to one 140-frame shot while five stretches had no render at all. This reads the
three files that already hold those facts and prints them side by side, so "finish the video" is a list.

**What it reads, each a named file handed in; nothing is found by a key.**

- `--shots`: the cuts of the video in source frames, and for every shot and every subject whether that
  subject is in it (`yes`, `no`, `unsure`) with who said so and from what. Schema `h3_shots/1`
  (`SHOTS_SCHEMA`); `data/<stem>/shots.json` is where a video's lives. A shot a subject is `unsure` on is a
  row of its own, since nothing can be planned on it. Not read here and read by `bench/shot_to_load.py`: on a
  subject of a shot `frames_on` and `frames_boxed` (the frames a person marked them on, and the frames a box
  of theirs stands on in the cast file; `bench/page_answers_to_cast.py --shots` writes both), and on a shot
  `order`, a person's list of its subjects in the order their trackers run.
- `--plan`: for each subject what is replaced by default (`face`, `face and hair`, `whole`) with who ruled it,
  how a delivery piece is known to be that subject's (`load_prefixes`), the table's own names for the
  subjects where they differ (`aliases`, as the delivery tool's `--alias`), and dated notes on a stretch
  (`departures`). Schema `h3_plan/1` (`PLAN_SCHEMA`). Not read here and read by `bench/shot_to_load.py`: a
  subject's `still` (the picture its face is taken from, under the server's input folder), `parts` (the part
  model's ticks for it) and, optionally, `hair_band` (a number, or `{"value": N, ...}` with where it is from).
- `--table`: the delivery table `bench/assemble_delivery.py` builds from; its two readers are imported from
  there, so the format has one definition.
- `--ledger` (optional): the render ledger (`bench/render_ledger.py --json`), the one inventory of this video's
  renders: tracked, made new with every delivery table, and knowing every copy of the source a render was loaded
  from. From it each render's file, name and frames, what its saved graph replaces (`whole subject` or `the wired
  parts`), whether a motion video was wired, and, where the ledger could read it from the build record and its
  capture, WHICH region (`face`, `face and hair`, `whole`): a piece is then held to the plan's region exactly, and
  only a render the ledger could not place that far is held to the graph's kind.
- `--renders` (optional, the older way; not with `--ledger`): the inventory read from each render's own saved graph
  by the review page's builder (`data/<stem>/poc/renders.json`). It was last made on 2026-10-11, knows two copies
  of the source and names none of the pieces laid since; kept for a file that exists, not made new. With either, each piece's region is
  read from its graph (`whole subject` or `the wired parts`), and the renders on a shot that the table does
  not use are listed, so a session sees that eleven renders of one shot exist before making a twelfth.
- `--verdicts` (optional): the owner's lines on renders, as `bench/page_answers_to_verdicts.py` copies them
  from the review page into `data/<stem>/verdicts/verdicts.jsonl`; each is printed under the render it is
  about, oldest first, unreworded, so the table carries the owner's eye beside the files' facts.
- `--gap-check` (optional; or `assembly.stack.coverage_json` in `<table_stem>.json`): the per-frame gap check
  (`bench/frame_coverage.py`), held by `table_sha256`, `span`, and `clip_check` to `--table`; at the top the JSON's
  `by_shot` `remake` and `capture` lists (worst shots first, in frame-subjects with gap spans), the four other run kinds
  (`carried_not_known`, `original_her_track_empty`, `left_out_by_a_look`, `seam`), any `failures`, and per row the
  frames where the original shows by kind (`original_while_tracked`, `emptied_inside`, `laid_but_barely_changed`,
  `region_far_below_its_own_size`; and, apart, `original_tracking_not_known`).

**What it prints, a row a shot and a subject present or unsure:** the frames, the subject, what the plan
wants, what the table lays there (a piece and its region, or the `source` line's own words, or nothing), and
one of the states in `STATES`, each a plain fact about the files and not a judgement of any render. "Present, nothing
to make under this plan" is a subject in the shot whose plan parts a person's lines (`seen_looks` in the shots file,
written by `bench/shots_look.py`) say are seen on no frame of it (`nothing_to_make`): not a gap and not a stopped load. Then
frames by state for each subject, and the pieces the table lays that no plan row claims. A render in the
table is "as planned" when its graph's region is the kind the plan names (parts for `face` and `face and
hair`; the graph does not say which parts, so with `--renders` face against face and hair is NOT read and the
row says so; with `--ledger` a render whose region the ledger names is held to it) and no frame of the shot shows the
original in the gap check (`original shows on some frames`, `partly, original shows`, or `original shows` when gap
frames exist); "not as planned" when they differ. A piece covering part of a shot says which frames.

**What it cannot see.** Whether a render is any good (the owner's eye on playback; the records under
`bench/results/`). Whether the parts a render replaced are the face or the face and hair (not in the saved
graph's summary). Whether a shot's cuts are right: the shots file carries the tracker's cuts and whoever
looked; a `note` on a shot is a person's doubt, printed beside it.

The frame numbers are the copy's, as the whole folder's are (`clip_folder.py`, "One clock").
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from frame_coverage import CAPTURE_GAP_KINDS, KIND_WORDS, OTHER_RUN_KINDS, REMAKE_GAP_KINDS, run_words  # noqa: E402

SHOTS_SCHEMA, PLAN_SCHEMA = "h3_shots/1", "h3_plan/1"
PRESENT = ("yes", "no", "unsure")
REGIONS = ("face", "face and hair", "whole")
RECIPE_CROPS = ("cranium", "head", "face", "torso")
RECIPE_OPEN_MOUTHS = ("inner outline", "capped", "outlined", "outlined over the lips")
RECIPE_CLOSED_MOUTHS = ("sculpted", "outlined")
#: What a saved graph's `replace` says, and which plan regions it is the kind of.
GRAPH_KIND = {"whole subject": ("whole",), "the wired parts": ("face", "face and hair")}
STATES = (
    "as planned",            # a table piece of this subject covers the shot, its region is the plan's kind, and 0 gap frames show the original
    "original shows on some frames",  # a piece covers the shot with the plan's kind, and some frames show the original in the gap check
    "original shows",        # a piece is laid with the plan's kind, and every frame of the shot shows the original in the gap check
    "not as planned",        # a piece covers it and its graph's region is the other kind
    "region not readable",   # a piece covers it and no renders inventory names it, so its region is not read
    "partly, as planned",    # a piece of this subject covers some frames of the shot and not the rest, with 0 gap frames; its kind as above
    "partly, original shows",  # a piece covers some frames of the shot with the plan's kind, and gap frames show the original
    "partly, not as planned",
    "partly, region not readable",
    "source, said why",      # a `source` line leaves the subject as the original here, with words
    "source, no line",       # the subject is here and the table says nothing about them on these frames
    "presence unsure",       # the shots file cannot say the subject is here, so nothing is planned
    "present, nothing to make under this plan",   # a person's lines (`seen_looks`) say none of the plan's parts is seen on any frame
)
#: What a look may say of a part on frames of a shot (`who[label].seen_looks`, written by `bench/shots_look.py`).
SEEN_IS = ("not seen", "seen")
REMAKE_GAP_KINDS_ALL = (
    "captured_not_rendered",
    "rendered_not_kept",
    "original_while_tracked",
    "emptied_inside",
    "laid_but_barely_changed",
    "region_far_below_its_own_size",
)
CAPTURE_GAP_KINDS_ALL = CAPTURE_GAP_KINDS + ("original_tracking_not_known",)
GAP_KINDS = REMAKE_GAP_KINDS_ALL + CAPTURE_GAP_KINDS_ALL


def _delivery():
    """`read_table` and `read_intents` from the delivery tool, imported from its file so the format has one home."""
    spec = importlib.util.spec_from_file_location("assemble_delivery", HERE / "assemble_delivery.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.read_table, mod.read_intents


def load_shots(path) -> dict:
    d = json.load(open(path))
    if d.get("schema") != SHOTS_SCHEMA:
        raise SystemExit(f"{path}: schema is {d.get('schema')!r}, not {SHOTS_SCHEMA!r}")
    shots = d.get("shots") or []
    last = None
    for s in shots:
        if s["first"] > s["last"] or (last is not None and s["first"] != last + 1):
            raise SystemExit(f"{path}: shot {s['first']}-{s['last']} does not follow {last} with no gap and no overlap")
        last = s["last"]
        for label in d["subjects"]:
            who = s["who"].get(label)
            if not who or who.get("present") not in PRESENT:
                raise SystemExit(f"{path}: shot {s['first']}-{s['last']}: `{label}` needs `present` in {PRESENT}")
            for look in who.get("seen_looks") or []:
                bad = [k for k in ("part", "why", "by", "at") if not str(look.get(k) or "").strip()]
                if bad or look.get("is") not in SEEN_IS or not (s["first"] <= int(look.get("first", -1)) <= int(look.get("last", -2)) <= s["last"]):
                    raise SystemExit(f"{path}: shot {s['first']}-{s['last']}: a look on `{label}` is not a part, frames inside the shot, `is` in "
                                     f"{SEEN_IS}, why, by and at: {look}")
    lo, hi = d["span"]
    if shots and (shots[0]["first"] != lo or shots[-1]["last"] != hi):
        raise SystemExit(f"{path}: the shots cover {shots[0]['first']}-{shots[-1]['last']}, the span says {lo}-{hi}")
    return d


def _validate_recipe_fields(path, label: str, where: str, rec: dict) -> None:
    if not isinstance(rec, dict) or not str(rec.get("why") or rec.get("by") or "").strip():
        raise SystemExit(f"{path}: `{label}`: `{where}` must be a dict with non-empty `why` or `by`")
    if "zoom_up_to" in rec and rec["zoom_up_to"] is not None:
        z = rec["zoom_up_to"]
        if isinstance(z, bool) or not isinstance(z, int | float) or not (2.0 <= float(z) <= 4.0):
            raise SystemExit(f"{path}: `{label}`: `{where}.zoom_up_to` must be a number in 2.0..4.0, not {z!r}")
    if "margin" in rec and rec["margin"] is not None:
        m = rec["margin"]
        if isinstance(m, bool) or not isinstance(m, int) or not (1 <= m <= 512):
            raise SystemExit(f"{path}: `{label}`: `{where}.margin` must be an integer in 1..512, not {m!r}")
    if "crop" in rec and rec["crop"] is not None and rec["crop"] not in RECIPE_CROPS:
        raise SystemExit(f"{path}: `{label}`: `{where}.crop` must be one of {RECIPE_CROPS}, not {rec['crop']!r}")
    if "open_mouth" in rec and rec["open_mouth"] is not None and rec["open_mouth"] not in RECIPE_OPEN_MOUTHS:
        raise SystemExit(f"{path}: `{label}`: `{where}.open_mouth` must be one of {RECIPE_OPEN_MOUTHS}, not {rec['open_mouth']!r}")
    if "closed_mouth" in rec and rec["closed_mouth"] is not None and rec["closed_mouth"] not in RECIPE_CLOSED_MOUTHS:
        raise SystemExit(f"{path}: `{label}`: `{where}.closed_mouth` must be one of {RECIPE_CLOSED_MOUTHS}, not {rec['closed_mouth']!r}")


def load_plan(path, subjects) -> dict:
    d = json.load(open(path))
    if d.get("schema") != PLAN_SCHEMA:
        raise SystemExit(f"{path}: schema is {d.get('schema')!r}, not {PLAN_SCHEMA!r}")
    for label, p in d["subjects"].items():
        if label not in subjects:
            raise SystemExit(f"{path}: `{label}` is not a subject of the shots file ({', '.join(subjects)})")
        if p.get("region") not in REGIONS:
            raise SystemExit(f"{path}: `{label}`: region must be one of {REGIONS}")
        if "recipe" in p and p["recipe"] is not None:
            _validate_recipe_fields(path, label, "recipe", p["recipe"])
        if "shot_overrides" in p and p["shot_overrides"] is not None:
            if not isinstance(p["shot_overrides"], list):
                raise SystemExit(f"{path}: `{label}`: `shot_overrides` must be a list of span dicts")
            for i, ov in enumerate(p["shot_overrides"]):
                if (
                    not isinstance(ov, dict)
                    or isinstance(ov.get("first"), bool)
                    or isinstance(ov.get("last"), bool)
                    or not isinstance(ov.get("first"), int)
                    or not isinstance(ov.get("last"), int)
                    or ov["first"] > ov["last"]
                ):
                    raise SystemExit(f"{path}: `{label}`: `shot_overrides[{i}]` needs integer `first <= last`: {ov!r}")
                _validate_recipe_fields(path, label, f"shot_overrides[{i}]", ov)
        if "restore" in p:
            ok = isinstance(p["restore"], list) and p["restore"] and str(p.get("restore_by") or "").strip()
            for x in p["restore"] if isinstance(p["restore"], list) else []:
                ok = ok and isinstance(x, dict) and isinstance(x.get("classes"), list) and x["classes"] and all(isinstance(c, str) and c for c in x["classes"]) \
                    and isinstance(x.get("first"), int) and isinstance(x.get("last"), int) and x["first"] <= x["last"]
            if not ok:
                # named frames only: a class map calls her eye makeup Eyeglass on a shot without glasses, and a telephone receiver
                # too (2026-10-14, the lead's rows 0294d, 1802b and 1575b), so a restore on every row would paste the original back
                raise SystemExit(f"{path}: `{label}`: `restore` is a list of {{\"classes\": [the part model's classes], \"first\": F, \"last\": L}}, each "
                                 "given back to the source on the pieces of this subject on those source frames only (bench/delivery_table.py), with "
                                 "`restore_by` saying who asked and why")
    for label in subjects:
        if label not in d["subjects"]:
            raise SystemExit(f"{path}: no plan for `{label}`")
    return d


def nothing_to_make(who: dict, wanted: dict, first: int, last: int) -> list[str] | None:
    """A person's lines that none of the plan's parts for a subject is seen on any frame of a shot, as said; None where
    they do not say so. THE OWNER'S WORD OF 2026-10-14 (about 17:05Z): a subject is IN a shot even where nothing is made
    of them, and a later plan (another part of them replaced) re-derives its loads from the same records. THE LEAD'S
    RULING: a person's line is needed; a part model's reading under a floor only marks a candidate (`bench/shot_parts.py`),
    since faces that are made sit under it. Read: the subject present `yes`, the plan names its `parts`, and for each of
    them the looks (`who[label].seen_looks`, in the order they were written, a later `seen` taking back an earlier `not
    seen`) leave every frame of the shot `not seen`."""
    parts = list(wanted.get("parts") or [])
    looks = sorted(who.get("seen_looks") or [], key=lambda x: x["at"])
    if who.get("present") != "yes" or not parts or not looks:
        return None
    whole = set(range(first, last + 1))
    for part in parts:
        unseen: set[int] = set()
        for x in looks:
            if x["part"] == part:
                span = set(range(max(int(x["first"]), first), min(int(x["last"]), last) + 1))
                unseen = unseen | span if x["is"] == "not seen" else unseen - span
        if unseen != whole:
            return None
    return [f"{x['part']} {x['is']} on {x['first']}-{x['last']}: {x['why']} ({x['by']}, {x['at']})" for x in looks if x["part"] in parts]


def alias(label: str, plan: dict) -> str:
    """A table's name for a subject (version a's `singer`) read through the plan's `aliases` to the plan's label."""
    return {k: v for k, v in plan.get("aliases", {}).items() if not k.startswith("_")}.get(label, label)


def subject_of(row: dict, plan: dict) -> tuple[str | None, str]:
    """Whose a table piece is: `subject=` on the row, else the load name's prefix from the plan's `load_prefixes`."""
    if row.get("subject"):
        return alias(row["subject"], plan), "the row's subject="
    name = Path(row["piece"]).name
    for label, prefixes in plan.get("load_prefixes", {}).items():
        for p in prefixes:
            if name.startswith(p):
                return label, f"the load name's prefix `{p}`"
    return None, "nothing names the subject"


def render_of(piece: str, renders: list[dict]) -> dict | None:
    for r in renders:
        if piece.endswith(r["file"]):
            return r
    return None


LEDGER_SCHEMA = "h3_render_ledger/"


def renders_of_ledger(ledger: dict) -> list[dict]:
    """The render ledger's renders as the rows `coverage` walks, with the region the ledger names where it is one of ours."""
    if not str(ledger.get("schema", "")).startswith(LEDGER_SCHEMA):
        raise SystemExit(f"--ledger: schema is {ledger.get('schema')!r}, not {LEDGER_SCHEMA}N: this is not bench/render_ledger.py's --json")
    out = []
    for r in ledger["renders"].values():
        graph, form, span = r.get("graph") or {}, r.get("form") or {}, r.get("span") or [None, None]
        if span[0] is None or span[1] is None:
            continue
        out.append({"file": r["file"], "name": r["name"], "first": int(span[0]), "last": int(span[1]),
                    "as_the_graph_says": {"replace": graph.get("replace")}, "motion_wired": str(graph.get("motion_reference", "")).startswith("a video I wire"),
                    "region_exact": form.get("region") if form.get("region") in REGIONS else None})
    return out


def read_renders(ledger: str | None, renders: str | None) -> list[dict]:
    """The renders `coverage` walks, from the ledger or from the older inventory; never from both."""
    if ledger and renders:
        raise SystemExit("--ledger and --renders are two inventories of the same renders: give one (the ledger is the one made new with every table)")
    if ledger:
        return renders_of_ledger(json.load(open(ledger)))
    if renders:
        d = json.load(open(renders))
        return d["renders"] if isinstance(d, dict) else d
    return []


def overlap(a, b, c, d) -> tuple[int, int] | None:
    lo, hi = max(a, c), min(b, d)
    return (lo, hi) if lo <= hi else None


def owner_said(render_file: str | None, verdicts: list[dict]) -> list[str]:
    """The owner's lines on a render from the verdicts record, oldest first, as `at: what they said`."""
    if not render_file:
        return []
    out = []
    for v in sorted((v for v in verdicts if v.get("render") and render_file.endswith(v["render"])), key=lambda v: v["at"]):
        said = v["chosen"] if v.get("what") == "choice" else str(v.get("words") or "").strip().replace("\n", " / ")
        where = f" at frame {v['frame']}" if v.get("frame") not in (None, "") else ""
        out.append(f"{v['at'][:16]}{where}: {said}")
    return out


def _in_repo(path: str, near: Path) -> Path:
    if Path(path).is_absolute():
        return Path(path)
    for anc in [*near.resolve().parents, HERE.parent]:
        if (anc / path).exists():
            return anc / path
    return HERE.parent / path


def table_span(TT: dict) -> tuple[int, int] | None:
    shots = list(TT.get("reading") or []) + list(TT.get("not_in_the_shot") or [])
    rr = shots or list(TT.get("rows") or [])
    return (min(int(r["first"]) for r in rr), max(int(r["last"]) for r in rr)) if rr else None


def read_gap_check(table_path: str | Path, gap_check_path: str | Path | None = None, rows: list[dict] | None = None) -> dict:
    """The gap check (`bench/frame_coverage.py --json`) named by the table's companion `.json` (`assembly.stack.coverage_json`),
    or by `--gap-check`, verified against the table `.json`'s rows, `table_sha256`, `span`, and `clip_check`."""
    t_json = Path(table_path) if str(table_path).endswith(".json") else Path(table_path).with_suffix(".json")
    if not t_json.is_file():
        raise SystemExit(f"--table {table_path}: companion table JSON {t_json} is missing (cannot verify table_sha256)")
    T = json.loads(t_json.read_text())
    if gap_check_path is not None:
        cov_rel = str(gap_check_path)
        cov_path = _in_repo(cov_rel, Path(table_path))
    else:
        cov_rel = ((T.get("assembly") or {}).get("stack") or {}).get("coverage_json")
        if not cov_rel:
            raise SystemExit(f"--table {table_path}: {t_json} names no gap check (assembly.stack.coverage_json is missing)")
        cov_path = _in_repo(str(cov_rel), t_json)
    if not cov_path.is_file():
        raise SystemExit(f"--table {table_path}: the gap check {cov_rel} is missing (looked at {cov_path})")
    cov = json.loads(cov_path.read_text())
    want_sha = hashlib.sha256(t_json.read_bytes()).hexdigest()
    got_sha = cov.get("table_sha256")
    if got_sha != want_sha:
        raise SystemExit(f"--table {table_path}: the gap check {cov_rel} has table_sha256 {got_sha!r}, not {t_json}'s {want_sha}")
    t_rows = T.get("rows")
    if rows is not None:
        if not isinstance(t_rows, list) or len(rows) != len(t_rows) or not all(
            int(x["first"]) == int(y.get("first", -1)) and int(x["last"]) == int(y.get("last", -2))
            and bool(y.get("file")) and (str(x["piece"]).endswith(str(y["file"])) or str(y["file"]).endswith(str(x["piece"])))
            for x, y in zip(rows, t_rows)
        ):
            raise SystemExit(f"--table {table_path}: table rows do not match companion JSON {t_json}")
    by_shot = cov.get("by_shot")
    if not isinstance(by_shot, dict) or not isinstance(by_shot.get("remake"), list) or not isinstance(by_shot.get("capture"), list):
        raise SystemExit(f"--table {table_path}: the gap check {cov_rel} has no by_shot remake and capture lists")
    t_span = table_span(T)
    c_span = cov.get("span")
    if c_span is not None and t_span is not None:
        if not isinstance(c_span, list) or len(c_span) != 2 or (int(c_span[0]), int(c_span[1])) != t_span:
            raise SystemExit(f"--table {table_path}: the gap check {cov_rel} is for span {c_span!r}, not this table's span {list(t_span)}")
    cc = cov.get("clip_check")
    if not isinstance(cc, dict) or not cc.get("file") or not cc.get("sha256"):
        raise SystemExit(f"--table {table_path}: the gap check {cov_rel} names no clip_check (file and sha256 required)")
    cc_path = _in_repo(str(cc["file"]), t_json)
    if not cc_path.is_file():
        raise SystemExit(f"--table {table_path}: the gap check {cov_rel}'s clip_check {cc['file']} is missing on disk")
    cc_sha = hashlib.sha256(cc_path.read_bytes()).hexdigest()
    if cc.get("sha256") != cc_sha:
        raise SystemExit(f"--table {table_path}: the gap check {cov_rel} has clip_check sha256 {cc.get('sha256')!r}, not {cc_path}'s {cc_sha}")
    cmd_parts = shlex.split(str(((T.get("assembly") or {}).get("command")) or ""))
    if "--out" not in cmd_parts[:-1]:
        raise SystemExit(f"--table {table_path}: assembly.command names no --out clip to tie clip_check to")
    out_arg = cmd_parts[cmd_parts.index("--out") + 1]
    want_cc = _in_repo(out_arg if out_arg.endswith(".check.json") else out_arg + ".check.json", t_json)
    if cc_path.resolve() != want_cc.resolve():
        raise SystemExit(f"--table {table_path}: the gap check {cov_rel} read clip_check {cc_path}, not the table's {want_cc}")
    return cov


def row_gap_check(gap_check: dict | None, first: int, last: int, label: str, plan: dict) -> dict | None:
    """For one shot and subject, the frames and runs where the original shows from `gap_check`, plus other runs (`OTHER_RUN_KINDS`)."""
    if gap_check is None:
        return None
    runs_here = []
    other_here = []
    by_kind: dict[str, int] = {}
    for r in gap_check.get("runs") or []:
        kind = r.get("kind")
        if alias(str(r.get("subject") or ""), plan) != label:
            continue
        r_shot = r.get("shot")
        if r_shot and [int(r_shot[0]), int(r_shot[1])] != [first, last]:
            continue
        both = overlap(first, last, int(r["first"]), int(r["last"]))
        if not both:
            continue
        lo, hi = both
        n = hi - lo + 1
        entry = {"frames": [lo, hi], "kind": kind, "cause": r.get("cause"), "pieces": list(r.get("pieces") or [])}
        if kind in GAP_KINDS:
            by_kind[kind] = by_kind.get(kind, 0) + n
            runs_here.append(entry)
        elif kind in OTHER_RUN_KINDS:
            other_here.append(entry)
    if not by_kind:
        for which, kinds in (("remake", REMAKE_GAP_KINDS_ALL), ("capture", CAPTURE_GAP_KINDS_ALL)):
            for x in (gap_check.get("by_shot") or {}).get(which) or []:
                subs = [alias(str(s), plan) for s in (x.get("subjects") or [])]
                if len(subs) == 1 and [int(x["shot"][0]), int(x["shot"][1])] == [first, last] and label in subs:
                    for k in kinds:
                        v = int((x.get("by_kind") or {}).get(k, 0))
                        if v:
                            by_kind[k] = by_kind.get(k, 0) + v
    ordered = {k: by_kind[k] for k in GAP_KINDS if by_kind.get(k)}
    return {"frames": sum(ordered.values()), "by_kind": ordered, "runs": runs_here, "other_runs": other_here}


def coverage(shots: dict, plan: dict, rows: list[dict], intents: list[dict], renders: list[dict],
             verdicts: list[dict] | None = None, gap_check: dict | None = None) -> dict:
    verdicts = verdicts or []
    span = shots["span"]
    for row in rows:
        if row["first"] < span[0] or row["last"] > span[1]:
            raise SystemExit(f"table row {row['first']}-{row['last']} `{row['piece']}` is outside the shots file's span {span}")
    pieces = []
    for row in rows:
        label, how = subject_of(row, plan)
        r = render_of(row["piece"], renders)
        pieces.append(dict(row, label=label, label_how=how, render=r, region_exact=(r or {}).get("region_exact"),
                           region=(r or {}).get("as_the_graph_says", {}).get("replace")))
    out = {"video": shots["video"], "span": span, "source_frames": shots.get("source_frames"), "rows": [],
           "unclaimed_pieces": [dict(first=p["first"], last=p["last"], piece=p["piece"], why=p["label_how"])
                                for p in pieces if p["label"] is None]}
    if gap_check is not None:
        out["gap_check"] = {"table": gap_check.get("table"), "table_sha256": gap_check.get("table_sha256"),
                            "by_shot": gap_check["by_shot"], "counts": gap_check.get("counts"),
                            "failures": list(gap_check.get("failures") or []), "runs": list(gap_check.get("runs") or [])}
    for s in shots["shots"]:
        for label in shots["subjects"]:
            who = s["who"][label]
            if who["present"] == "no":
                continue
            wanted = plan["subjects"][label]
            notes = [d for d in plan.get("departures", []) if d["subject"] == label
                     and overlap(s["first"], s["last"], d["first"], d["last"])]
            row = {"first": s["first"], "last": s["last"], "frames": s["last"] - s["first"] + 1, "subject": label,
                   "present": who["present"], "present_how": who.get("how", ""), "present_by": who.get("by", ""),
                   "wanted": wanted["region"], "wanted_by": wanted.get("by", ""),
                   "notes": [f"{d['first']}-{d['last']}: {d['note']} ({d.get('by', '')})" for d in notes],
                   "shot_note": s.get("note", ""), "laid": [], "left": [], "others": [],
                   "nothing_to_make": nothing_to_make(who, wanted, s["first"], s["last"])}
            gc = row_gap_check(gap_check, s["first"], s["last"], label, plan)
            if gc is not None:
                row["gap_check"] = gc
            mine = [p for p in pieces if p["label"] == label and overlap(s["first"], s["last"], p["first"], p["last"])]
            for p in mine:
                lo, hi = overlap(s["first"], s["last"], p["first"], p["last"]) or (0, -1)
                row["laid"].append({"frames": [lo, hi], "piece": p["piece"], "region": p["region"], "region_exact": p["region_exact"],
                                    "named_by": p["label_how"],
                                    "crosses_a_cut": p["first"] < s["first"] or p["last"] > s["last"],
                                    "owner_said": owner_said(p["piece"], list(verdicts))})
            for it in intents:
                both = overlap(s["first"], s["last"], it["first"], it["last"])
                if alias(it["subject"], plan) == label and both:
                    lo, hi = both
                    row["left"].append({"frames": [lo, hi], "why": it["why"]})
            for r in renders:
                if not overlap(s["first"], s["last"], r["first"], r["last"]):
                    continue
                rl, _ = subject_of({"piece": r["file"]}, plan)
                if rl != label:
                    continue
                row["others"].append({"name": r["name"], "frames": [r["first"], r["last"]],
                                      "region": r.get("region_exact") or r.get("as_the_graph_says", {}).get("replace"),
                                      "motion": bool(r.get("motion_wired")),
                                      "in_table": any(p["render"] is r for p in mine),
                                      "owner_said": owner_said(r["file"], list(verdicts))})
            row["state"] = state_of(row)
            out["rows"].append(row)
    out["frames_by_state"] = totals(out["rows"])
    return out


def state_of(row: dict) -> str:
    if row["present"] == "unsure":
        return "presence unsure"
    covered = set()
    for p in row["laid"]:
        covered.update(range(p["frames"][0], p["frames"][1] + 1))
    whole = set(range(row["first"], row["last"] + 1))
    if row["laid"]:
        # a piece whose region the ledger names is held to the plan's region itself; any other to the graph's kind
        held = [p["region_exact"] == row["wanted"] if p.get("region_exact") else None if p["region"] is None
                else row["wanted"] in GRAPH_KIND.get(p["region"], ()) for p in row["laid"]]
        kind = "region not readable" if None in held else "as planned" if all(held) else "not as planned"
        gap_n = int((row.get("gap_check") or {}).get("frames") or 0)
        if kind == "as planned" and gap_n > 0:
            if gap_n >= row["frames"]:
                return "original shows"
            return "partly, original shows" if covered != whole else "original shows on some frames"
        return f"partly, {kind}" if covered != whole else kind
    if row.get("nothing_to_make"):
        return "present, nothing to make under this plan"
    return "source, said why" if row["left"] else "source, no line"


def totals(rows: list[dict]) -> dict:
    t = defaultdict(lambda: defaultdict(int))
    for r in rows:
        gap_n = min(int((r.get("gap_check") or {}).get("frames") or 0), int(r["frames"])) if r.get("laid") else 0
        if gap_n > 0:
            t[r["subject"]]["original shows"] += gap_n
            if r["frames"] - gap_n > 0:
                base_state = ("as planned" if r["state"] == "original shows on some frames"
                              else "partly, as planned" if r["state"] == "partly, original shows" else r["state"])
                t[r["subject"]][base_state] += r["frames"] - gap_n
        else:
            t[r["subject"]][r["state"]] += r["frames"]
    return {s: dict(v) for s, v in t.items()}


def frames(a, b) -> str:
    return f"{a}-{b}"


def span_frames(a: int, b: int) -> str:
    return f"{int(a)}" if int(a) == int(b) else f"{int(a)}-{int(b)}"


def fmt_shot_spans_plain(x: dict, shot_runs: list[dict]) -> str:
    subs_order = list(dict.fromkeys(str(s) for s in (x.get("subjects") or [r.get("subject") for r in shot_runs if r.get("subject")])))
    groups: dict[tuple[tuple[int, int], ...], list[str]] = {}
    for sub in subs_order:
        sp = tuple(dict.fromkeys(sorted((int(r["first"]), int(r["last"])) for r in shot_runs if str(r.get("subject") or "") == sub)))
        if sp:
            groups.setdefault(sp, []).append(sub)
    if not groups:
        return ""
    return " on " + "; ".join(f"{', '.join(g_subs)}: {', '.join(span_frames(lo, hi) for lo, hi in sp)}" for sp, g_subs in groups.items())


def print_report(out: dict) -> None:
    print(f"{out['video']}: the shots file covers {frames(*out['span'])}"
          + (f" of {out['source_frames']} frames in the source; nothing past {out['span'][1]} has a shot, a plan or a render here"
             if out.get("source_frames") else ""))
    exact = any(p.get("region_exact") for r in out["rows"] for p in r["laid"])
    print("A row a shot and a subject in it (or unsure). The state is about the files, not about any render's quality; `as planned` "
          + ("holds a piece to the plan's region itself where the ledger names it (face, face and hair, whole) and to the graph's kind (parts or whole) where it does not."
             if exact else "reads the graph's kind only (parts or whole), never face against face and hair."))
    print()
    if out.get("gap_check"):
        gc = out["gap_check"]
        by_shot = gc.get("by_shot") or {}
        all_runs = gc.get("runs") or []
        failures = gc.get("failures") or []
        print(f"Where the original still shows (from the gap check, table_sha256 {str(gc.get('table_sha256') or '')[:12]}, worst shots first):")
        if failures:
            print(f"  FAILED GAP CHECK ({len(failures)} failure(s)):")
            for fl in failures:
                print(f"    {fl}")
        for which, title, kinds, all_kinds in (
            ("remake", "remake (a laid piece fails, or no piece where tracked)", REMAKE_GAP_KINDS, REMAKE_GAP_KINDS_ALL),
            ("capture", "capture (original, tracking not known)", CAPTURE_GAP_KINDS, CAPTURE_GAP_KINDS_ALL),
        ):
            entries = by_shot.get(which) or []
            words = "; ".join(f"{k} = {KIND_WORDS[k]}" for k in kinds)
            print(f"  {title}: {len(entries)} shot(s), {sum(int(x.get('frames', 0)) for x in entries)} frame-subject(s) [{words}]")
            for x in entries:
                s0, s1 = int(x["shot"][0]), int(x["shot"][1])
                bk = x.get("by_kind") or {}
                k_str = ", ".join(f"{k}: {int(bk.get(k, 0))}" for k in all_kinds if int(bk.get(k, 0)))
                subs = ", ".join(str(s) for s in (x.get("subjects") or []))
                s_runs = [r for r in all_runs if r.get("kind") in all_kinds and r.get("shot") and [int(r["shot"][0]), int(r["shot"][1])] == [s0, s1]]
                span_str = fmt_shot_spans_plain(x, s_runs)
                print(f"    {frames(s0, s1)} ({subs}): {int(x['frames'])} frame-subject(s) ({k_str}){span_str}")
        print("  Other runs from the gap check:")
        for k in OTHER_RUN_KINDS:
            k_runs = [r for r in all_runs if r.get("kind") == k]
            n_fs = sum(int(r["last"]) - int(r["first"]) + 1 for r in k_runs)
            count_str = f"{len(k_runs)} seam(s)" if k == "seam" else f"{n_fs} frame-subject(s)"
            where = (" on " + ", ".join(f"{span_frames(int(r['first']), int(r['last']))} ({r.get('subject')})" for r in k_runs)) if k_runs else ""
            print(f"    {k} ({KIND_WORDS[k]}): {count_str}{where}")
        print()
    for r in out["rows"]:
        print(f"{frames(r['first'], r['last'])}  {r['frames']:4d} f  {r['subject']:<7} wanted: {r['wanted']:<14} STATE: {r['state']}")
        if r["present"] == "unsure" or r["present_how"]:
            print(f"    here: {r['present']}; {r['present_how']} ({r['present_by']})")
        for line in r.get("nothing_to_make") or []:
            print(f"    NOTHING TO MAKE UNDER THIS PLAN, a person's look: {line}")
        for p in r["laid"]:
            cut = "; crosses a cut" if p["crosses_a_cut"] else ""
            print(f"    laid {frames(*p['frames'])}: {Path(p['piece']).name}  region: {p.get('region_exact') or p['region'] or 'not read (no renders entry)'}"
                  f"  subject by {p['named_by']}{cut}")
            for said in p.get("owner_said", []):
                print(f"        the owner, {said}")
        for it in r["left"]:
            print(f"    source {frames(*it['frames'])}: {it['why']}")
        gc_row = r.get("gap_check")
        if gc_row and gc_row.get("frames"):
            bk = gc_row.get("by_kind") or {}
            k_str = ", ".join(f"{k} {v}" for k, v in bk.items() if v)
            spans_str = "; ".join(f"{span_frames(*x['frames'])} {x['kind']}" + (f" ({x['cause']})" if x.get("cause") else "") for x in gc_row.get("runs") or [])
            print(f"    original shows ({gc_row['frames']} f: {k_str})" + (f": {spans_str}" if spans_str else ""))
        if gc_row and gc_row.get("other_runs"):
            other_str = "; ".join(f"{span_frames(*x['frames'])} {x['kind']}" + (f" ({x['cause']})" if x.get("cause") else "") for x in gc_row["other_runs"])
            print(f"    other gap-check runs: {other_str}")
        for n in r["notes"]:
            print(f"    note {n}")
        if r["shot_note"]:
            print(f"    shot note: {r['shot_note']}")
        others = [o for o in r["others"] if not o["in_table"]]
        if others:
            print(f"    other renders of this subject here, not in the table: "
                  + "; ".join(f"{o['name']} {frames(*o['frames'])} {o['region']}{' +motion' if o['motion'] else ''}" for o in others))
            for o in others:
                for said in o.get("owner_said", []):
                    print(f"        the owner on {o['name']}, {said}")
    print()
    print("Frames by state, a subject at a time (a subject's shots where they are present or unsure):")
    for label, t in out["frames_by_state"].items():
        print(f"  {label}: " + ", ".join(f"{k} {v}" for k, v in sorted(t.items(), key=lambda kv: -kv[1])))
    if out["unclaimed_pieces"]:
        print()
        print("Pieces in the table whose subject nothing names (not counted above):")
        for p in out["unclaimed_pieces"]:
            print(f"  {frames(p['first'], p['last'])} {Path(p['piece']).name}: {p['why']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--shots", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--ledger", help="the render ledger (bench/render_ledger.py --json): the one inventory, made new with every table; without it or --renders no region is read")
    ap.add_argument("--renders", help="the older inventory read from saved graphs by the review page's builder; not with --ledger")
    ap.add_argument("--verdicts", help="the owner's lines on renders (bench/page_answers_to_verdicts.py's record); printed under each render")
    ap.add_argument("--gap-check", help="the gap check's JSON (bench/frame_coverage.py --json); read from the table's companion .json when left out")
    ap.add_argument("--json", help="write the same rows as JSON here")
    a = ap.parse_args()
    read_table, read_intents = _delivery()
    shots = load_shots(a.shots)
    plan = load_plan(a.plan, shots["subjects"])
    renders = read_renders(a.ledger, a.renders)
    verdicts = [json.loads(t) for t in Path(a.verdicts).read_text().splitlines() if t.strip()] if a.verdicts else []
    table_txt = str(Path(a.table).with_suffix(".txt")) if str(a.table).endswith(".json") and Path(a.table).with_suffix(".txt").is_file() else a.table
    rows = read_table(table_txt)
    gap_check = read_gap_check(a.table, a.gap_check, rows)
    out = coverage(shots, plan, rows, read_intents(table_txt), renders, verdicts, gap_check)
    out["read_from"] = {"shots": a.shots, "plan": a.plan, "table": a.table, "ledger": a.ledger, "renders": a.renders, "verdicts": a.verdicts,
                        "gap_check": (gap_check or {}).get("table") and str(a.gap_check or ((json.loads((Path(a.table) if str(a.table).endswith(".json") else Path(a.table).with_suffix(".json")).read_text()).get("assembly") or {}).get("stack") or {}).get("coverage_json"))}
    print_report(out)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwritten: {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
