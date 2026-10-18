#!/usr/bin/env python3
"""Per frame, per shot and subject: the furthest step reached on the 4-step ladder, naming the files.

    <python> bench/frame_coverage.py --table data/<...>/t_<...>.json [--span A-B] [--json OUT.json] [--md OUT.md]

**Why it exists** (the owner's order of 2026-10-16 and 2026-10-17, `docs/wiki/decisions.md`). In the kitchen a person laid as
swapped turned back into the original halfway through the shot, and nothing in the ledger or the pages said so: the table
counted a gap against the whole shot, not against the frames she is in, two renders of one person meeting were not
flagged, and the ledger read only the frames a person's `drop=` emptied, not the floor's or the grade's. The method is
the flip survey's of 2026-10-15 (the part model plus our floor emptied every flip it found), ported, not imported.

**The 4-step ladder** (`shots[*].step`, `shots[*].step_name`, `shots[*].files`, and the top-level lists). Each frame of each
shot and subject gets the furthest step reached, naming the files:

1. **nothing captured** (`no_capture`, `"no capture"`): nothing under `data/` (tied or untied) holds her SAM 3.1 track on
   these frames (`subjects/<label>/masks__<by>.npz`), naming any capture covering the frames without her track;
2. **captured, not rendered** (`captured_not_rendered`, `"captured, not rendered"`): the capture under `data/` (tied or
   untied) that holds her track on these frames (`track_px > 0`), with no render covering them, naming the capture and
   `<capture>/subjects/<label>/masks__<by>.npz`;
3. **rendered, not kept** (`rendered_not_kept`, `"rendered, not kept"`): the rendered clips covering these frames for her
   that are not in the table, and their windows, from the build records `render_ledger.py` reads
   (`*_job/*.render.built.json` and `*.render.queued.json`, plus `ledger["renders"]` when handed in);
4. **kept and laid** (`kept_and_laid`, `"kept and laid"`): in the table — carried (with held frames noted), or
   `emptied_inside` (`"emptied inside a piece"`, by cause: `drop=`, `floor`, `grade`, `nothing saved`,
   `left_as_the_source_with_no_mask`, `across a cut`, or `not in the record`, joined by `+`), or
   `laid_but_barely_changed` (`"laid piece barely changed its region"`), or `region_far_below_its_own_size`
   (`"carried region far below its own size"`).

**One home of the check's words.** `KIND_WORDS`, `CHIP_WORDS`, `run_words()`, `chip_words_of()`, and `clip_gap_line()`
define the check's own words for every run and per-clip summary line; `stack_delivery.py`, `review_pages.py`, and
`shot_coverage.py` read them from here and never keep their own wording tables.

**Seams** (`seams`): two different laid renders of one person meeting on one shot, with both render days.

**Left out by a look** (the lead's ruling of 2026-10-16, as built). Frames the shots record says the subject's part is
NOT SEEN on (`who[label].seen_looks`, entered by `bench/shots_look.py`; a later `seen` takes them back) are their own
kind (`left_out_by_a_look`, `"left out by a look"`), with who looked and why. A laid piece's own job `left_out` is NOT
taken as a look: it is said beside the run it covers (`a_jobs_left_out_says`).

**Laid but barely changed** (the lead's order of 2026-10-16): the assembler records what each piece changed on each frame
(`<clip>.mp4.check.json`, `regions.pieces.<piece>.changed_px_per_frame`). Each frame a piece carried a mask on, its changed
px are set against the px of the region its run carried (`runs/<run>/region.npz`), both counted at the canvas. A share under
`BARELY_CHANGED` is a run of `laid_but_barely_changed`. The gap check is refused when the clip check is missing or its
pieces and frames are not the table's rows.

**Region far below its own size** (the lead's order of 2026-10-16): a frame whose carried region is under
`FAR_BELOW_ITS_OWN_SIZE` of the piece's own median carried px is `region_far_below_its_own_size`.

**By shot**, worst first in each of two lists (`by_shot`): (a) `remake` (`REMAKE_GAP_KINDS`: `captured_not_rendered`,
`rendered_not_kept`, `emptied_inside`, `laid_but_barely_changed`, `region_far_below_its_own_size`), which a render or lay
answers; (b) `capture` (`CAPTURE_GAP_KINDS`: `no_capture`), which a capture answers.

**What it cannot see.** The picture (`CANNOT`). Green or red only for a row whose `run=` does not resolve or a file it
names that is missing.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from statistics import median

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent if (HERE.parent / "data").is_dir() or not (Path.cwd() / "data").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(HERE))
from piece_gaps import day, runs  # noqa: E402

#: set bits of a byte, to count a packed mask's px without unpacking it
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)
CANNOT = ("the picture: a frame whose mask was carried can still come out as the original (no per-frame likeness of the delivered "
          "video is read); whether a tracked person's face is visible (track px > 0 is SAM 3.1's pixels on her, printed so 'barely in' "
          "shows; whether her face shows on such frames is a playback question); a two-subject run's carried mask is both subjects' union; "
          "a face-and-hair run whose face emptied while its hair was carried counts as carried (the px are the union); a restore= that gives "
          "the source back at assembly inside a carried mask is not read; a subject the shots file says is not in a shot is not walked (the "
          "table's not_in_the_shot); an older capture's subject named by an alias is not matched as untied; each frame and person gets the "
          "furthest step reached (1: no capture; 2: captured, not rendered; 3: rendered, not kept; 4: kept and laid), naming the files")

#: Canonical run kinds for `by_shot`, the stacked view (`stack_delivery.py`), the review pages (`review_pages.py`), and `shot_coverage.py`.
REMAKE_GAP_KINDS = (
    "captured_not_rendered",
    "rendered_not_kept",
    "emptied_inside",
    "laid_but_barely_changed",
    "region_far_below_its_own_size",
)
CAPTURE_GAP_KINDS = ("no_capture",)
OTHER_RUN_KINDS = ("carried_not_known", "original_her_track_empty", "left_out_by_a_look", "seam")
CLIP_GAP_KINDS = (
    "no_capture",
    "captured_not_rendered",
    "rendered_not_kept",
    "emptied_inside",
    "left_out_by_a_look",
    "laid_but_barely_changed",
    "region_far_below_its_own_size",
    "original_while_tracked",
    "original_tracking_not_known",
)
LOST_KINDS = ("captured_not_rendered", "rendered_not_kept", "original_while_tracked")

#: The gap check's own words for each run kind (the one home of these words: read by `stack_delivery.py`, `review_pages.py`, and `shot_coverage.py`).
KIND_WORDS = {
    "no_capture": "no capture",
    "captured_not_rendered": "captured, not rendered",
    "rendered_not_kept": "rendered, not kept",
    "emptied_inside": "emptied inside a piece",
    "laid_but_barely_changed": "laid piece barely changed its region",
    "region_far_below_its_own_size": "carried region far below its own size",
    "left_out_by_a_look": "left out by a look",
    "original_her_track_empty": "original, her track is empty (0 px)",
    "carried_not_known": "tracked, whether the laid piece carried a mask is not known",
    "seam": "seam where two laid pieces of the same person meet",
    # Legacy kind aliases so older or hand-built coverage dicts resolve to the check's words:
    "original_while_tracked": "captured, not rendered",
    "original_tracking_not_known": "no capture",
}

#: Chip headlines for `stack_delivery.py`, using the gap check's own words in uppercase (`SEAM` shortened so 10 seam chips fit a 288 px test frame).
CHIP_WORDS = {
    "no_capture": "NO CAPTURE",
    "captured_not_rendered": "CAPTURED, NOT RENDERED",
    "rendered_not_kept": "RENDERED, NOT KEPT",
    "emptied_inside": "EMPTIED INSIDE A PIECE",
    "laid_but_barely_changed": "LAID PIECE BARELY CHANGED ITS REGION",
    "region_far_below_its_own_size": "CARRIED REGION FAR BELOW ITS OWN SIZE",
    "left_out_by_a_look": "LEFT OUT BY A LOOK",
    "original_her_track_empty": "ORIGINAL, HER TRACK IS EMPTY (0 PX)",
    "carried_not_known": "MASK NOT KNOWN",
    "seam": "SEAM",
    "original_while_tracked": "ORIGINAL WHILE TRACKED",
    "original_tracking_not_known": "THE ORIGINAL: NO PIECE MADE FOR THESE FRAMES (no capture here, so tracking is not shown)",
}
CHIP_NO_TRACK_WORDS = "THE ORIGINAL: NO PIECE MADE FOR THESE FRAMES, AND THE CAPTURE THERE HOLDS NO TRACK OF HER"


def run_words(run: dict) -> str:
    """The gap check's own words for a run: `run['words']` when stamped on the run, else `KIND_WORDS[run['kind']]`."""
    if run.get("words"):
        return str(run["words"])
    return KIND_WORDS[str(run["kind"])]


def chip_words_of(run: dict) -> str:
    """The stacked view's chip text for a run, in the gap check's own words (`run['chip_words']` or `run['words']`) plus `cause`."""
    if run.get("chip_words"):
        return str(run["chip_words"])
    kind = str(run["kind"])
    raw_cause = str(run.get("cause") or "")
    cause = raw_cause.removesuffix(
        ": the original: no piece made for these frames (no capture here, so tracking is not shown)"
    ).removesuffix(
        ": the original: no piece made for these frames, and the capture there holds no track of her"
    ).removesuffix(": tracking not known")
    if run.get("words") and kind != "seam":
        head = str(run["words"]).upper()
    elif kind == "original_tracking_not_known" and "holds no track of her" in raw_cause:
        head = CHIP_NO_TRACK_WORDS
    else:
        head = CHIP_WORDS[kind]
    return head + (f" ({cause})" if cause else "")


def clip_gap_line(cov: dict | None, first: int, last: int, subjects: list[str] | None = None) -> str:
    """The gap check's counts for a clip's own frames `[first, last]`: per person, each kind (`no capture`; `captured, not rendered`;
    `rendered, not kept`; `emptied inside a piece`; `left out by a look`; `laid piece barely changed its region`;
    `carried region far below its own size`) with frame counts and spans, in the gap check's own words; empty string when whole."""
    if not cov:
        return ""
    matched = [
        r for r in (cov.get("runs") or [])
        if r.get("kind") in CLIP_GAP_KINDS
        and int(r["first"]) <= last
        and int(r["last"]) >= first
        and (subjects is None or str(r.get("subject") or "") in subjects)
    ]
    if not matched:
        return ""
    subs = [s for s in subjects if any(str(r.get("subject") or "") == s for r in matched)] if subjects else sorted({str(r.get("subject") or "") for r in matched})
    per_person = []
    for sub in subs:
        sub_runs = [r for r in matched if str(r.get("subject") or "") == sub]
        by_words: dict[str, list[tuple[int, int]]] = {}
        ordered = sorted(sub_runs, key=lambda r: (CLIP_GAP_KINDS.index(r["kind"]) if r["kind"] in CLIP_GAP_KINDS else 99, int(r["first"]), int(r["last"])))
        for r in ordered:
            w = run_words(r)
            lo, hi = max(int(r["first"]), first), min(int(r["last"]), last)
            spans = by_words.setdefault(w, [])
            if spans and lo <= spans[-1][1] + 1:
                spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
            else:
                spans.append((lo, hi))
        parts = []
        for w, spans in by_words.items():
            n_f = sum(hi - lo + 1 for lo, hi in spans)
            span_txt = ", ".join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in spans)
            parts.append(f"{w}: {n_f} ({span_txt})")
        per_person.append(f"{sub}: {'; '.join(parts)}")
    return f"Gaps on {first}-{last} — " + " · ".join(per_person)


#: The share of the carried region a laid piece must change on a frame, under which the frame is `laid_but_barely_changed`.
#: **Measured** starting value, set between the owner's 806 and 807 on the 2003Z cut at the corrected scale
#: (`fun_0000_6394_joined_2026-10-16T2003Z_from_laid.mp4.check.json`; both counts at the canvas, Gemini's D17, 2026-10-16):
#: lead_fp_0805d's 806 is 0.226 (1810 of 8008 px; the owner: it "doesn't swap") and 807 0.404 (3735 of 9235; "seems to swap"),
#: its 828-832 0.511 to 0.566 (no complaint). It was first set on shares understated by the loader crop's area over the canvas's
#: (1.93 on this video). The shares are printed so a reader sees how far under.
BARELY_CHANGED = 0.3

#: The share of a piece's own median carried px under which a frame's carried region is `region_far_below_its_own_size`: the
#: piece changed what it carried, and what it carried was a scrap of her (her face hidden), so the original shows round it.
#: **Measured** starting value, on the same cut (2642 carried frames, 2026-10-16): lead_fp_0805d's 805 is 0.041 (486 px against
#: its median 11853; the owner: "finds her... ear... at 805") and lead_fp_1419d's 1448 0.077; lead_fp_0751c's 751 is 0.100, and
#: thirteen frames fall under 0.15, eleven of them lead_fp_1419d's and third_fp_1419d's. No break in the shares: a line, not a
#: gap, and the px and the median are printed beside it.
FAR_BELOW_ITS_OWN_SIZE = 0.1

#: Inherited from `bench/render_ledger.py::NUMBERED` and `NOT_A_RENDER`: how a build/queue record names a rendered clip.
import re as _re  # noqa: E402
NUMBERED = _re.compile(r"_\d{5}_?\.(?:mp4|mov|mkv)$")
NOT_A_RENDER = ("_with_mask", "_zoomed", "_window_", "_probe")


class Refused(Exception):
    """The gap check is not run: the clip's check it reads is missing or is not of this table."""


def px(packed: np.ndarray) -> np.ndarray:
    """Px a frame of a packed bool mask [n, h, ceil(w/8)]."""
    return POP[packed.reshape(len(packed), -1)].sum(1, dtype=np.int64)


def rel(path: Path) -> str:
    """A path from the repo root when it is under it, else whole (a check's temporary folder)."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def frames_of(spans) -> set[int]:
    """A manifest's frame list, as spans [[a, b], ...] or single frames."""
    out: set[int] = set()
    for x in spans or []:
        if isinstance(x, (list, tuple)):
            out.update(range(int(x[0]), int(x[1]) + 1))
        else:
            out.add(int(x))
    return out


class Failure(Exception):
    """A hard failure: a row whose `run=` does not resolve, or a file it names that is missing."""


class Capture:
    """One capture folder's records, read once."""

    def __init__(self, folder: Path):
        if not (folder / "manifest.json").is_file():
            raise Failure(f"{folder}: no manifest.json")
        self.folder = folder
        self.m = json.loads((folder / "manifest.json").read_text())
        self.first, self.n = int(self.m["first_frame"]), int(self.m["frames"])
        self._track: dict = {}
        self._run: dict = {}

    def sighting(self, label: str) -> dict | None:
        s = next((s for s in self.m.get("subjects", []) if s["label"] == label), None)
        if s is None or not s.get("sightings"):
            return None
        return next((x for x in s["sightings"] if (self.folder / "subjects" / label / f"masks__{x['by']}.npz").is_file()), s["sightings"][0])

    def mask_file(self, label: str) -> str | None:
        """The capture's track mask file for `label`, named as `<capture>/subjects/<label>/masks__<by>.npz`."""
        seen = self.sighting(label)
        if seen is None:
            return None
        name = f"masks__{seen['by']}.npz"
        path = self.folder / "subjects" / label / name
        return f"{self.folder.name}/subjects/{label}/{name}" if path.is_file() else None

    def region_file(self, name: str) -> str | None:
        """The capture's carried region file for `name`, named as `<capture>/runs/<name>/region.npz`."""
        rz = self.folder / "runs" / name / "region.npz"
        return f"{self.folder.name}/runs/{name}/region.npz" if rz.is_file() else None

    def track(self, label: str) -> dict | None:
        """Her SAM 3.1 track px and covered flags on this capture's frames, and the saved part's px; None when the
        capture has no masks file for her."""
        if label not in self._track:
            seen = self.sighting(label)
            path = self.folder / "subjects" / label / f"masks__{seen['by']}.npz" if seen else None
            if path is None or not path.is_file():
                self._track[label] = None
            else:
                z = np.load(path)
                self._track[label] = {"px": px(z["track"]), "covered": z["covered"].astype(bool),
                                      "part_px": px(z["parts"]) if "parts" in z.files else None, "file": rel(path),
                                      "mask_file": f"{self.folder.name}/subjects/{label}/masks__{seen['by']}.npz"}
        return self._track[label]

    def track_at(self, label: str, f: int) -> int | None:
        t = self.track(label)
        k = f - self.first
        if t is None or not 0 <= k < self.n or not t["covered"][k]:
            return None
        return int(t["px"][k])

    def run(self, name: str) -> dict:
        if name not in self._run:
            entry = next((r for r in self.m.get("runs", []) if r["name"] == name), None)
            if entry is None:
                raise Failure(f"{self.folder.name}: no run {name!r} in its manifest")
            rz = self.folder / "runs" / name / "region.npz"
            if not rz.is_file():
                raise Failure(f"{rz}: missing")
            z = np.load(rz)
            self._run[name] = {"entry": entry, "carried_px": px(z["carried"]), "read": z["read"].astype(bool),
                               "file": rel(rz), "region_file": f"{self.folder.name}/runs/{name}/region.npz"}
        return self._run[name]

    def grades(self, label: str) -> dict[int, dict]:
        p = self.folder / "subjects" / label / "part_grades.json"
        return {int(g["source_frame"]): g for g in json.loads(p.read_text())["frames"]} if p.is_file() else {}


def why_emptied(cap: Capture, run: dict, label: str, f: int, grades: dict[int, dict]) -> str:
    """Why a run carried no mask for her on frame f, every cause the records give, in the order the chain makes the
    part; "not in the record" when none says."""
    seen = cap.sighting(label) or {}
    entry = run["entry"]
    causes = []
    if f in frames_of(entry.get("left_as_the_source_across_a_cut")):
        causes.append("a cut")
    if f in frames_of(seen.get("parts_emptied_on_source_frames")):
        causes.append("drop=")
    g = grades.get(f)
    if g and g.get("did") == "emptied":
        causes.append("floor" if str(g.get("why") or "").startswith("too small") else "grade")
    elif f in frames_of(seen.get("parts_emptied_by_grade_on_source_frames")):
        causes.append("floor or grade (part_grades.json does not say which)")
    t = cap.track(label)
    k = f - cap.first
    if t is not None and t["part_px"] is not None and 0 <= k < cap.n and t["part_px"][k] == 0:
        causes.append("nothing saved")
    if f in frames_of(entry.get("left_as_the_source_with_no_mask")):
        causes.append("the run's no-mask list")
    return "+".join(causes) or "not in the record"


def emptied(cap: Capture, run_name: str, label: str, span: list[int]) -> dict:
    """The frames inside `span` on which the run carried no mask for `label`, each with its cause; the rule the ledger's
    `left_by_the_load` reads too (one home). Frames the capture did not read are listed apart, not as emptied."""
    run = cap.run(run_name)
    grades = cap.grades(label)
    out, unread = {}, []
    for f in range(max(span[0], cap.first), min(span[1], cap.first + cap.n - 1) + 1):
        k = f - cap.first
        if not run["read"][k]:
            unread.append(f)
        elif run["carried_px"][k] == 0:
            out[f] = why_emptied(cap, run, label, f, grades)
    return {"frames": out, "not_read": unread, "read_from": run["file"]}


def resolve(row: dict, caps: dict[str, Capture], data: Path | None = None) -> tuple[Capture, str]:
    if not row.get("run") or ":" not in row["run"]:
        raise Failure(f"row {row.get('name')} {row.get('first')}-{row.get('last')}: no run= to tie it to a capture")
    folder, name = row["run"].rsplit(":", 1)
    if folder not in caps:
        candidates = [Path(folder)] if Path(folder).is_absolute() else (
            ([data / folder.removeprefix("data/"), data.parent / folder, data / folder] if data is not None else [])
            + [REPO / folder, Path.cwd() / folder]
        )
        path = next((c.resolve() for c in candidates if c.is_dir()), (REPO / folder).resolve())
        if not path.is_dir():
            raise Failure(f"row {row.get('name')}: run={row['run']} names no folder")
        caps[folder] = Capture(path)
    cap = caps[folder]
    cap.run(name)
    return cap, name


def untied_captures(clips: set[str], tied: set[str], data: Path) -> list[Capture]:
    """Every capture of these clips under `data/` that no row ties to, read so Step 2 (`captured, not rendered`) finds
    her track whether the capture is tied to a row in the table or not. Blocked or superseded folders are skipped."""
    out = []
    for mp in sorted(data.glob("*/manifest.json")):
        if mp.parent.name in tied or any(w in mp.parent.name.lower() for w in ("blocked", "superseded")):
            continue
        try:
            m = json.loads(mp.read_text())
            if clips and m.get("clip") not in clips or m.get("status") in ("blocked", "superseded") or m.get("blocked"):
                continue
            out.append(Capture(mp.parent))
        except (ValueError, KeyError, TypeError, Failure):
            continue
    return out


def _clean_rel(path: str | Path | None, data: Path) -> str | None:
    if not path:
        return None
    p = Path(path)
    for base in (REPO, Path.cwd(), data):
        try:
            return str(p.resolve().relative_to(base.resolve()))
        except ValueError:
            pass
    parts = p.parts
    for marker in ("Video", "data"):
        if marker in parts:
            return "/".join(parts[parts.index(marker):])
    return p.name


def unkept_renders_of(table: dict, data: Path, ledger: dict | None, clips: set[str]) -> list[dict]:
    """The rendered clips and their windows from the build records `render_ledger.py` reads (`*_job/*.render.built.json`
    and `*.render.queued.json` under `data/`, plus `ledger['renders']` when handed in) that are not in the table."""
    laid_files = {str(r.get("file") or "") for r in table.get("rows", [])}
    laid_names = {str(r.get("name") or "") for r in table.get("rows", [])}
    out: list[dict] = []
    seen_keys: set[tuple] = set()

    # 1. Build records under data/*_job/ (the exact pairs render_ledger.load_records reads, plus built records with no queue file yet)
    built_paths = sorted(set(data.glob("*_job/*.render.built.json")) | {
        q.with_name(q.name.replace(".queued.json", ".built.json"))
        for q in data.glob("*_job/*.render.queued.json")
        if q.with_name(q.name.replace(".queued.json", ".built.json")).is_file()
    })
    for built_p in built_paths:
        queued_p = built_p.with_name(built_p.name.replace(".built.json", ".queued.json"))
        try:
            b = json.loads(built_p.read_text())
            q = json.loads(queued_p.read_text()) if queued_p.is_file() else {}
        except ValueError:
            continue
        load = b.get("load") or {}
        first, frames_n = load.get("first"), load.get("frames")
        if first is None or not frames_n:
            continue
        if b.get("capture") and clips:
            cap_p = Path(b["capture"]) if Path(b["capture"]).is_absolute() else (REPO / b["capture"])
            if not cap_p.is_dir() and (data / Path(b["capture"]).name).is_dir():
                cap_p = data / Path(b["capture"]).name
            if (cap_p / "manifest.json").is_file():
                try:
                    m_clip = json.loads((cap_p / "manifest.json").read_text()).get("clip")
                    if m_clip and m_clip not in clips:
                        continue
                except ValueError:
                    pass
        r_span = [int(first), int(first) + int(frames_n) - 1]
        subj = load.get("subject")
        subjects = [s for s in ([subj] + [x.get("subject") for x in (b.get("subjects") or []) if isinstance(x, dict)]) if s]
        if not subjects:
            continue
        q_files = [str(f) for f in (q.get("files") or []) if NUMBERED.search(str(f)) and not any(w in str(f) for w in NOT_A_RENDER)]
        load_name = str(load.get("name") or built_p.name.removesuffix(".render.built.json"))
        if not q_files and b.get("output_folder"):
            out_dir = Path(b["output_folder"])
            if out_dir.is_dir():
                found = sorted(p.name for p in out_dir.glob(f"{load_name}_*.mp4") if NUMBERED.search(p.name) and not any(w in p.name for w in NOT_A_RENDER))
                q_files = [f"{_clean_rel(out_dir, data)}/{fn}" for fn in found] if found else [f"{_clean_rel(out_dir, data)}/{load_name}.mp4"]
            else:
                q_files = [f"{_clean_rel(out_dir, data)}/{load_name}.mp4"]
        clip_file = q_files[0] if q_files else load_name
        if clip_file in laid_files or load_name in laid_names or any(f in laid_files for f in q_files):
            continue
        win_val = ((b.get("settings") or {}).get("window") or {}).get("value") or load.get("window")
        ctx_val = ((b.get("settings") or {}).get("context") or {}).get("value") or load.get("context")
        win_folder = _clean_rel(b.get("windows_folder"), data) if b.get("windows_folder") else None
        win_files = [str(w) for w in (q.get("windows_files") or []) if str(w).endswith(".mp4") and not any(x in str(w) for x in ("_with_mask", "_zoomed"))]
        if not win_files and b.get("windows_folder") and Path(b["windows_folder"]).is_dir():
            win_files = sorted(p.name for p in Path(b["windows_folder"]).glob("*_window_*.mp4") if not any(x in p.name for x in ("_with_mask", "_zoomed")))
        if win_files:
            windows = list(win_files)
        elif win_folder and win_val:
            windows = [f"{win_folder} (window {win_val}" + (f", context {ctx_val}" if ctx_val is not None else "") + ")"]
        elif win_val:
            windows = [f"window {win_val}" + (f", context {ctx_val}" if ctx_val is not None else "")]
        elif win_folder:
            windows = [win_folder]
        else:
            windows = []
        built_rel = _clean_rel(built_p, data)
        queued_rel = _clean_rel(queued_p, data) if queued_p.is_file() else None
        key = (clip_file, tuple(r_span), tuple(subjects))
        if key not in seen_keys:
            seen_keys.add(key)
            out.append({
                "name": load_name,
                "clip": clip_file,
                "files": q_files or [clip_file],
                "subjects": subjects,
                "span": r_span,
                "window": win_val,
                "context": ctx_val,
                "windows_folder": win_folder,
                "windows": windows,
                "built_file": built_rel,
                "queued_file": queued_rel,
            })

    # 2. Renders in the ledger when handed in
    renders_raw = (ledger or {}).get("renders") or []
    ledger_list = list(renders_raw.values()) if isinstance(renders_raw, dict) else list(renders_raw)
    for r in ledger_list:
        if not isinstance(r, dict) or not r.get("span"):
            continue
        r_span = [int(r["span"][0]), int(r["span"][1])]
        subjects = [s for s in (r.get("subjects") or ([r.get("subject")] if r.get("subject") else [])) if s]
        if not subjects:
            continue
        clip_file = str(r.get("file") or r.get("name") or "")
        load_name = str(r.get("name") or Path(clip_file).stem)
        if clip_file in laid_files or load_name in laid_names or r.get("laid_in_the_table"):
            continue
        recs = r.get("records") or []
        rec0 = recs[-1] if recs else {}
        win_val = rec0.get("window") or (r.get("graph") or {}).get("window")
        ctx_val = rec0.get("context") or (r.get("graph") or {}).get("context")
        win_folder = _clean_rel(r.get("windows_folder") or rec0.get("windows_folder"), data)
        win_files = list(r.get("windows") or rec0.get("windows") or [])
        if not win_files and win_val:
            win_files = [f"window {win_val}" + (f", context {ctx_val}" if ctx_val is not None else "")]
        built_rel = _clean_rel(rec0.get("built_file"), data)
        queued_rel = _clean_rel(rec0.get("queued_file"), data)
        key = (clip_file, tuple(r_span), tuple(subjects))
        if key not in seen_keys:
            seen_keys.add(key)
            out.append({
                "name": load_name,
                "clip": clip_file,
                "files": [clip_file],
                "subjects": subjects,
                "span": r_span,
                "window": win_val,
                "context": ctx_val,
                "windows_folder": win_folder,
                "windows": win_files,
                "built_file": built_rel,
                "queued_file": queued_rel,
            })
    return out


def left_out_by_looks(table: dict, ledger: dict | None, shots: dict | None, data: Path | None = None) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], dict]]:
    """Two maps of (subject, frame) -> {who, why, from}.

    `looks`: the frames the shots record says the subject's part is NOT SEEN on (`who[label].seen_looks`,
    `bench/shots_look.py`; a later `seen` takes an earlier `not seen` back). These are their own kind, "left out by a
    look", never a gap (the lead's ruling of 2026-10-16).

    `jobs`: the frames a laid piece's own job left out (`job.left_out`, from the job file the ledger's record of the
    render names), carried beside those frames as context and NOT taken out of list 1: a job's left_out is sometimes a
    look ("his face is behind his blurred hand") and sometimes "left for a later load ... not ruled out" (lead_fp_0852b
    on 604-845, the kitchen's own gaps), and only free text tells them apart. A look that rules frames the original
    is entered into the shots record."""
    import re
    looks: dict[tuple[str, int], dict] = {}
    jobs: dict[tuple[str, int], dict] = {}
    renders_raw = (ledger or {}).get("renders") or {}
    renders_by_file = renders_raw if isinstance(renders_raw, dict) else {r.get("file"): r for r in renders_raw if isinstance(r, dict) and r.get("file")}
    for row in table.get("rows", []):
        r = renders_by_file.get(row.get("file")) or {}
        rec = next((c for c in reversed(r.get("records") or []) if c.get("job_file")), None)
        if rec is None:
            continue
        jf = Path(rec["job_file"]) if Path(rec["job_file"]).is_absolute() else (REPO / rec["job_file"])
        if not jf.is_file() and data is not None:
            for cand in (data / str(rec["job_file"]).removeprefix("data/"), data.parent / rec["job_file"], data / rec["job_file"], Path.cwd() / rec["job_file"]):
                if cand.is_file():
                    jf = cand
                    break
        if not jf.is_file():
            continue
        job = json.loads(jf.read_text()).get("job") or {}
        m = re.search(r"written by (\S+) at (\d{4}-\d\d-\d\dT[\d:]+Z?)", str(job.get("note") or ""))
        who = f"the job of {row.get('name')}" + (f", written by {m.group(1)} at {m.group(2)}" if m else "")
        for x in job.get("left_out") or []:
            for f in range(int(x["first"]), int(x["first"]) + int(x["frames"])):
                jobs[(x["subject"], f)] = {"who": who, "why": x.get("why"), "from": rec["job_file"]}
    for shot in (shots or {}).get("shots", []):
        for label, w in (shot.get("who") or {}).items():
            for look in w.get("seen_looks") or []:
                for f in range(int(look["first"]), int(look["last"]) + 1):
                    if look.get("is") == "not seen":
                        looks[(label, f)] = {"who": look.get("by"), "why": look.get("why"), "from": "the shots record's seen_looks"}
                    elif look.get("is") == "seen":
                        looks.pop((label, f), None)
    return looks, jobs


def named_in(table: dict, pattern: str, data: Path | None = None) -> Path | None:
    import re
    m = re.search(pattern, str(table.get("made") or ""))
    if not m:
        return None
    rel = m.group(1)
    for base in ([REPO, Path.cwd()] + ([data.parent, data] if data is not None else [])):
        if (base / rel).is_file():
            return (base / rel).resolve()
    return None


def written_by_render(table: dict) -> dict[str, str]:
    out = {}
    for r in table.get("reading", []):
        for s in r.get("stands", []):
            if s.get("written"):
                out[s["render"]] = s["written"]
    return out


def check_of(table: dict, path: Path | None = None) -> tuple[Path, dict]:
    """The clip's check (`<clip>.mp4.check.json`, the assembler's): the one the table's own assembly command writes (its
    `--out`), or `path`. Refused when it is missing, or when its pieces and frames are not the table's rows."""
    if path is None:
        cmd = shlex.split(str((table.get("assembly") or {}).get("command") or ""))
        if "--out" not in cmd[:-1]:
            raise Refused("the table names no assembled clip (its assembly command has no --out): give the clip's check with --check")
        path = Path(cmd[cmd.index("--out") + 1] + ".check.json")
    path = Path(path)
    if not path.is_file():
        raise Refused(f"{path}: missing. The gap check reads what each piece changed from the clip's check, so it is not run without "
                      "it: assemble the table first (its assembly command), or name the check with --check")
    check = json.loads(path.read_text())
    left = [(str(r["file"]), int(r["first"]), int(r["last"])) for r in table["rows"]]
    strays = []
    for x in check.get("table") or []:
        p, a, b = str(x["piece"]), int(x["frames"][0]), int(x["frames"][1])
        i = next((i for i, (f, fa, fb) in enumerate(left) if (fa, fb) == (a, b) and (p == f or p.endswith("/" + f.lstrip("/")))), None)
        if i is None:
            strays.append(f"{Path(p).name} {a}-{b}")
        else:
            left.pop(i)
    if strays or left:
        raise Refused(f"{path} is not of this table: pieces in the check and not the table: {strays or 'none'}; rows of the table not in "
                      f"the check: {[f'{Path(f).name} {a}-{b}' for f, a, b in left] or 'none'}")
    return path, check


def changed_of(check: dict, row: dict) -> dict[int, int]:
    """The px the assembler's `changed` called changed on each frame of a row's piece, from the clip's check."""
    piece = next((p for p in check["regions"]["pieces"] if p == row["file"] or p.endswith("/" + row["file"].lstrip("/"))), None)
    if piece is None:
        raise Refused(f"the clip's check holds no changed px for {row['name']} ({row['file']})")
    v = check["regions"]["pieces"][piece]
    a, b = (int(x) for x in v["frames"])
    if len(v["changed_px_per_frame"]) != b - a + 1:
        raise Refused(f"the clip's check holds {len(v['changed_px_per_frame'])} changed counts for {row['name']} over {a}-{b}: not one a frame")
    return {a + k: int(x) for k, x in enumerate(v["changed_px_per_frame"])}


def _format_render_summary(r: dict) -> str:
    w_str = ", ".join(r.get("windows") or []) or (r.get("windows_folder") or (f"window {r['window']}" if r.get("window") else "no windows named"))
    b_str = f"; from {r['built_file']}" if r.get("built_file") else ""
    return f"{r['clip']} [{r['span'][0]}-{r['span'][1]}] (windows: {w_str}{b_str})"


def coverage(table: dict, span: list[int] | None = None, data: Path | None = None, ledger: Path | dict | None = None,
             shots: Path | dict | None = None, check: Path | None = None) -> dict:
    check_path, clip_check = check_of(table, check)
    caps: dict[str, Capture] = {}
    failures = []
    tied, failed = [], []
    for row in table["rows"]:
        try:
            cap, name = resolve(row, caps, data)
            tied.append((row, cap, name))
        except Failure as e:
            failures.append(str(e))
            failed.append(row)
    if data is None:
        default_data = REPO / "data" if (REPO / "data").is_dir() or not (Path.cwd() / "data").is_dir() else (Path.cwd() / "data").resolve()
        data = next((c.folder.parent for c in caps.values()), default_data)
    written = written_by_render(table)
    clips_set = {c.m.get("clip") for c in caps.values() if c.m.get("clip")}
    tied_names = {c.folder.name for c in caps.values()}
    untied_caps = untied_captures(clips_set, tied_names, data)
    ledger_path = ledger if isinstance(ledger, Path) else (None if isinstance(ledger, dict) else named_in(table, r"from the ledger `([^`]+)`", data))
    ledger_data = ledger if isinstance(ledger, dict) else (json.loads(ledger_path.read_text()) if ledger_path and ledger_path.is_file() else None)
    shots_path = shots if isinstance(shots, Path) else (None if isinstance(shots, dict) else named_in(table, r"`([^`]*shots\.json)`", data))
    shots_data = shots if isinstance(shots, dict) else (json.loads(shots_path.read_text()) if shots_path and shots_path.is_file() else None)
    looked, job_left = left_out_by_looks(table, ledger_data, shots_data, data=data)
    unkept_renders = unkept_renders_of(table, data, ledger_data, clips_set)

    shots_out = []
    no_capture_list, captured_not_rendered_list, rendered_not_kept_list, kept_and_laid_list = [], [], [], []
    list1, list2, empty, carried_unknown, looked_out, barely, small = [], [], [], [], [], [], []

    for rd in table.get("reading", []):
        a, b, label = int(rd["first"]), int(rd["last"]), rd["subject"]
        if span and (b < span[0] or a > span[1]):
            continue
        present = rd.get("present")                                 # "yes" or "unsure": both are read, and said
        lo, hi = (max(a, span[0]), min(b, span[1])) if span else (a, b)
        in_shot = [(r, c, n) for r, c, n in tied if r["last"] >= a and r["first"] <= b]
        mine = [(r, c, n) for r, c, n in in_shot if r["subject"] == label]
        failed_mine = [r for r in failed if r.get("subject") == label and r["last"] >= a and r["first"] <= b]
        empt, changed_px, own_size = {}, {}, {}
        for r, c, n in mine:
            e = emptied(c, n, label, [max(r["first"], lo), min(r["last"], hi)])
            empt[id(r)] = e
            changed_px[id(r)] = changed_of(clip_check, r)
            own = c.run(n)
            sizes = [int(own["carried_px"][f - c.first]) for f in range(r["first"], r["last"] + 1)
                     if 0 <= f - c.first < c.n and own["read"][f - c.first] and own["carried_px"][f - c.first] > 0]
            own_size[id(r)] = float(median(sizes)) if sizes else None
        cols = {
            "frames": [], "step": [], "step_name": [], "files": [],
            "track_px": [], "track_from": [], "piece": [], "carried_px": [], "changed_share": [], "state": [],
        }
        row_meta: list[dict] = []
        for f in range(lo, hi + 1):
            piece = next(((r, c, n) for r, c, n in mine if r["first"] <= f <= r["last"]), None)
            # the piece's capture, then the shot's, then any capture a row of another shot ties to, then any untied capture of the clip
            order = ([piece] if piece else []) + [x for x in in_shot if x is not piece] + [x for x in tied if x not in in_shot]
            tpx, tfrom, tmask, ttied = None, None, None, False
            for _, c, _ in order:
                v = c.track_at(label, f)
                if v is not None:
                    tpx, tfrom, tmask, ttied = v, c.folder.name, c.mask_file(label), True
                    if v > 0:
                        break
            tied_px = tpx
            if tpx is None or tpx == 0:
                for uc in untied_caps:
                    uv = uc.track_at(label, f)
                    if uv is not None and (tpx is None or uv > 0):
                        tpx, tfrom, tmask, ttied = uv, uc.folder.name, uc.mask_file(label), False
                        if uv > 0:
                            break
            all_covering = [c.folder.name for _, c, _ in order if c.first <= f < c.first + c.n] + [
                uc.folder.name for uc in untied_caps if uc.first <= f < uc.first + uc.n
            ]
            without_track = sorted({f"{cn}/manifest.json" for cn in all_covering})
            if tpx is None:
                tfrom = ("nothing under data/ covers this frame" if not all_covering
                         else "a capture covers this frame and holds no track of her on it")
            unkept_here = [u for u in unkept_renders if label in u["subjects"] and u["span"][0] <= f <= u["span"][1]]
            carried, share = None, None
            bad = next((r for r in failed_mine if r["first"] <= f <= r["last"]), None)
            if piece is None and bad is not None:
                step, step_name = 4, "kept_and_laid"
                state = f"not known: the laid piece {bad.get('name')} has a run= that does not resolve"
                f_files = {"piece": bad.get("name"), "piece_file": bad.get("file"), "capture": None, "region_file": None, "mask_file": tmask}
            elif piece is None and (label, f) in looked:
                step = 3 if unkept_here else (2 if (tpx is not None and tpx > 0) else 1)
                step_name = "rendered_not_kept" if step == 3 else ("captured_not_rendered" if step == 2 else "no_capture")
                state = "left out by a look: " + str(looked[(label, f)]["why"])
                f_files = {"captures": [tfrom] if tpx is not None else [], "mask_files": [tmask] if tmask else [],
                           "renders": unkept_here if unkept_here else []}
            elif piece is None:
                if unkept_here:
                    step, step_name = 3, "rendered_not_kept"
                    state = "rendered, not kept: " + ", ".join(u["clip"] for u in unkept_here)
                    f_files = {
                        "renders": unkept_here,
                        "clips": [u["clip"] for u in unkept_here],
                        "windows": [w for u in unkept_here for w in u["windows"]],
                        "built_files": [u["built_file"] for u in unkept_here if u.get("built_file")],
                        "queued_files": [u["queued_file"] for u in unkept_here if u.get("queued_file")],
                        "captures": [tfrom] if tpx is not None else [],
                        "mask_files": [tmask] if tmask else [],
                    }
                elif tpx is not None and tpx > 0:
                    step, step_name = 2, "captured_not_rendered"
                    state = "original: no piece laid"
                    f_files = {"captures": [tfrom], "mask_files": [tmask] if tmask else [], "tied": ttied}
                elif tpx == 0:
                    step, step_name = 2, "original_her_track_empty"
                    state = "original: no piece laid"
                    f_files = {"captures": [tfrom], "mask_files": [tmask] if tmask else [], "tied": ttied}
                else:
                    step, step_name = 1, "no_capture"
                    state = "original: no piece laid"
                    f_files = {"captures": [], "mask_files": [], "captures_without_track": without_track}
            else:
                step, step_name = 4, "kept_and_laid"
                r, c, n = piece
                run = c.run(n)
                f_files = {
                    "piece": r["name"],
                    "piece_file": r.get("file"),
                    "capture": c.folder.name,
                    "region_file": c.region_file(n),
                    "mask_file": c.mask_file(label) or tmask,
                }
                k = f - c.first
                if not 0 <= k < c.n:
                    state = "not known: the piece's capture does not cover this frame"
                elif not run["read"][k]:
                    state = "not known: the capture did not read this frame of the run"
                else:
                    carried = int(run["carried_px"][k])
                    if carried == 0:
                        state = "original: emptied inside the piece (" + empt[id(r)]["frames"].get(f, "not in the record") + ")"
                    else:
                        share = changed_px[id(r)].get(f, 0) / carried
                        if own_size[id(r)] and carried < FAR_BELOW_ITS_OWN_SIZE * own_size[id(r)]:
                            state = f"region far below its own size (its median {int(own_size[id(r)])} px)"
                        elif share < BARELY_CHANGED:
                            state = "laid but barely changed"
                        elif f in frames_of((c.sighting(label) or {}).get("parts_held_on_source_frames")):
                            state = "carried (held from a neighbouring frame)"
                        else:
                            state = "carried"
            for key, v in (("frames", f), ("step", step), ("step_name", step_name), ("files", f_files),
                           ("track_px", tpx), ("track_from", tfrom), ("piece", piece[0]["name"] if piece else None),
                           ("carried_px", carried), ("changed_share", None if share is None else round(share, 4)), ("state", state)):
                cols[key].append(v)
            row_meta.append({"tmask": tmask, "ttied": ttied, "tied_px": tied_px, "without": without_track, "unkept": unkept_here, "files": f_files})
        shots_out.append({"shot": [a, b], "subject": label, "present": present, **cols})

        # Contiguous runs of Step 4 (kept and laid) for `kept_and_laid`
        cur4 = None
        for i, f in enumerate(cols["frames"]):
            if cols["piece"][i] is not None:
                st = cols["state"][i]
                ff = row_meta[i]["files"]
                k4 = (cols["piece"][i], st, ff.get("piece_file"), ff.get("region_file"))
                if cur4 is not None and cur4["key"] == k4 and f == cur4["f1"] + 1:
                    cur4["f1"] = f
                else:
                    if cur4 is not None:
                        kept_and_laid_list.append({
                            "shot": [a, b], "subject": label, "present": present,
                            "frames": [cur4["f0"], cur4["f1"]], "count": cur4["f1"] - cur4["f0"] + 1,
                            "step": 4, "kind": "kept_and_laid", "words": "kept and laid",
                            "piece": cur4["piece"], "state": cur4["state"], "files": cur4["files"],
                        })
                    cur4 = {"key": k4, "f0": f, "f1": f, "piece": cols["piece"][i], "state": st, "files": ff}
            elif cur4 is not None:
                kept_and_laid_list.append({
                    "shot": [a, b], "subject": label, "present": present,
                    "frames": [cur4["f0"], cur4["f1"]], "count": cur4["f1"] - cur4["f0"] + 1,
                    "step": 4, "kind": "kept_and_laid", "words": "kept and laid",
                    "piece": cur4["piece"], "state": cur4["state"], "files": cur4["files"],
                })
                cur4 = None
        if cur4 is not None:
            kept_and_laid_list.append({
                "shot": [a, b], "subject": label, "present": present,
                "frames": [cur4["f0"], cur4["f1"]], "count": cur4["f1"] - cur4["f0"] + 1,
                "step": 4, "kind": "kept_and_laid", "words": "kept and laid",
                "piece": cur4["piece"], "state": cur4["state"], "files": cur4["files"],
            })

        # Gap and issue runs across the 4-step ladder
        cur = None

        def close():
            if cur is None:
                return
            pxs = [x for x in cur["px"] if x is not None]
            item = {"shot": [a, b], "subject": label, "present": present, "frames": [cur["f0"], cur["f1"]], "count": cur["f1"] - cur["f0"] + 1,
                    "piece": cur["piece"], "shot_has_a_piece_of_her": bool(mine), "state": cur["state"], "track_from": sorted(cur["from"]),
                    "step": cur["step"]}
            said = [job_left[(label, f_)] for f_ in range(cur["f0"], cur["f1"] + 1) if (label, f_) in job_left]
            if cur["kind"] == "captured_not_rendered":
                item["kind"] = "captured_not_rendered"
                item["words"] = KIND_WORDS["captured_not_rendered"]
                item["tied"] = cur["tied"]
                item["mask_files"] = sorted(cur["masks"])
                item["files"] = {"captures": sorted(cur["from"]), "mask_files": sorted(cur["masks"]), "tied": cur["tied"]}
                item["track_px"] = {"least": min(pxs), "median": int(median(pxs)), "most": max(pxs)}
                if said:
                    item["a_jobs_left_out_says"] = {k: said[0][k] for k in ("who", "why", "from")} | {"frames": len(said)}
                captured_not_rendered_list.append(item)
                list1.append(item)
            elif cur["kind"] == "rendered_not_kept":
                item["kind"] = "rendered_not_kept"
                item["words"] = KIND_WORDS["rendered_not_kept"]
                item["tied"] = cur["tied"]
                item["tied_tracked_frames"] = sum(1 for tp in cur["tied_px"] if tp is not None and tp > 0)
                item["tied_empty_track_frames"] = sum(1 for tp in cur["tied_px"] if tp == 0)
                item["untied_frames"] = sum(1 for tp in cur["tied_px"] if tp is None)
                r_list = cur["renders"]
                item["renders"] = r_list
                item["clips"] = [u["clip"] for u in r_list]
                item["windows"] = [w for u in r_list for w in u["windows"]]
                item["built_files"] = [u["built_file"] for u in r_list if u.get("built_file")]
                item["queued_files"] = [u["queued_file"] for u in r_list if u.get("queued_file")]
                item["mask_files"] = sorted(cur["masks"])
                item["files"] = {
                    "renders": r_list,
                    "clips": item["clips"],
                    "windows": item["windows"],
                    "built_files": item["built_files"],
                    "queued_files": item["queued_files"],
                    "captures": sorted(cur["from"]),
                    "mask_files": sorted(cur["masks"]),
                }
                item["track_px"] = {"least": min(pxs), "median": int(median(pxs)), "most": max(pxs)} if pxs else {"least": 0, "median": 0, "most": 0}
                if said:
                    item["a_jobs_left_out_says"] = {k: said[0][k] for k in ("who", "why", "from")} | {"frames": len(said)}
                rendered_not_kept_list.append(item)
                list1.append(item)
            elif cur["kind"] == "emptied_inside":
                item["kind"] = "emptied_inside"
                item["words"] = KIND_WORDS["emptied_inside"]
                item["tied"] = True
                item["files"] = cur["files"]
                item["track_px"] = {"least": min(pxs), "median": int(median(pxs)), "most": max(pxs)} if pxs else {"least": 0, "median": 0, "most": 0}
                if said:
                    item["a_jobs_left_out_says"] = {k: said[0][k] for k in ("who", "why", "from")} | {"frames": len(said)}
                list1.append(item)
            elif cur["kind"] == "no_capture":
                item["kind"] = "no_capture"
                item["words"] = KIND_WORDS["no_capture"]
                without = sorted(cur["without"])
                why_str = ("not known: " + cur["state"].removeprefix("not known: ")) if cur["state"].startswith("not known") else str(cur["why"] or "nothing under data/ covers this frame")
                item["why"] = why_str + (f" ({', '.join(without)})" if without else "")
                item["tracking"] = item["why"]
                item["files"] = {"captures": [], "mask_files": [], "captures_without_track": without}
                no_capture_list.append(item)
            elif cur["kind"] == "untracked":
                item["kind"] = "original_her_track_empty"
                item["words"] = KIND_WORDS["original_her_track_empty"]
                item["tied"] = cur["tied"]
                item["files"] = cur["files"]
                empty.append(item)
            elif cur["kind"] == "looked_out":
                item["kind"] = "left_out_by_a_look"
                item["words"] = KIND_WORDS["left_out_by_a_look"]
                item["files"] = cur["files"]
                item.update({k: looked[(label, cur["f0"])][k] for k in ("who", "why", "from")})
                looked_out.append(item)
            elif cur["kind"] == "carried_not_known":
                item["kind"] = "carried_not_known"
                item["words"] = KIND_WORDS["carried_not_known"]
                item["files"] = cur["files"]
                item["track_px"] = {"least": min(pxs), "median": int(median(pxs)), "most": max(pxs)}
                carried_unknown.append(item)
            elif cur["kind"] == "barely":
                sh = cur["share"]
                item["kind"] = "laid_but_barely_changed"
                item["words"] = KIND_WORDS["laid_but_barely_changed"]
                item["files"] = cur["files"]
                item["changed_share"] = {"least": min(sh), "median": round(median(sh), 4), "most": max(sh), "under": BARELY_CHANGED}
                barely.append(item)
            elif cur["kind"] == "small":
                item["kind"] = "region_far_below_its_own_size"
                item["words"] = KIND_WORDS["region_far_below_its_own_size"]
                item["files"] = cur["files"]
                item["carried_px"] = cur["carried"]
                item["its_median_px"] = int(cur["state"].rsplit("median ", 1)[1].split(" ")[0])
                item["changed_share"] = cur["share"]
                small.append(item)

        for i, f in enumerate(cols["frames"]):
            st, tpx, step_i = cols["state"][i], cols["track_px"][i], cols["step"][i]
            rm = row_meta[i]
            if st == "laid but barely changed":
                kind = "barely"
            elif st.startswith("region far below its own size"):
                kind = "small"
            elif st.startswith("left out by a look"):
                kind = "looked_out"
            elif st.startswith("not known"):
                kind = "no_capture" if tpx is None else "carried_not_known"
            elif st.startswith("rendered, not kept"):
                kind = "rendered_not_kept"
            elif not st.startswith("original"):
                kind = None
            elif cols["piece"][i] is not None:
                kind = "emptied_inside" if (tpx is not None and tpx > 0) else ("untracked" if tpx == 0 else "emptied_inside")
            elif tpx is None:
                kind = "no_capture"
            elif tpx > 0:
                kind = "captured_not_rendered"
            else:
                kind = "untracked"
            why = cols["track_from"][i] if tpx is None else None
            r_clips = tuple(u["clip"] for u in rm["unkept"])
            key = (kind, cols["piece"][i], st, why, r_clips, rm["ttied"])
            if cur is not None and cur["key"] == key and f == cur["f1"] + 1:
                cur["f1"] = f
                cur["px"].append(tpx)
                cur["tied_px"].append(rm["tied_px"])
                cur["share"].append(cols["changed_share"][i])
                cur["carried"].append(cols["carried_px"][i])
                if cols["track_from"][i] and tpx is not None:
                    cur["from"].add(cols["track_from"][i])
                if rm["tmask"]:
                    cur["masks"].add(rm["tmask"])
                cur["without"].update(rm["without"])
            else:
                close()
                cur = None if kind is None else {
                    "key": key, "kind": kind, "step": step_i, "f0": f, "f1": f,
                    "piece": cols["piece"][i], "state": st, "why": why,
                    "px": [tpx], "tied_px": [rm["tied_px"]], "share": [cols["changed_share"][i]], "carried": [cols["carried_px"][i]],
                    "from": set() if tpx is None else {cols["track_from"][i]} - {None},
                    "masks": {rm["tmask"]} if rm["tmask"] else set(),
                    "without": set(rm["without"]),
                    "tied": rm["ttied"],
                    "renders": rm["unkept"],
                    "files": rm["files"],
                }
        close()
        # list 2: two different laid renders of her meeting inside this shot
        laid = sorted((r for r, _, _ in mine), key=lambda r: r["first"])
        for x, y in zip(laid, laid[1:]):
            if y["first"] == x["last"] + 1 and x["file"] != y["file"] and lo <= x["last"] and y["first"] <= hi:
                list2.append({"shot": [a, b], "subject": label, "at": [x["last"], y["first"]],
                              "before": {"piece": x["name"], "file": x.get("file"), "written": written.get(x["name"]), "day": day(written.get(x["name"]))},
                              "after": {"piece": y["name"], "file": y.get("file"), "written": written.get(y["name"]), "day": day(written.get(y["name"]))}})
    list3 = [x for x in list1 if x["piece"] is not None]
    runs_out = flat(list1, list2, no_capture_list, empty, written, carried_unknown, looked_out, barely, small)
    return {
        "failures": failures,
        "shots": shots_out,
        "no_capture": no_capture_list,
        "captured_not_rendered": captured_not_rendered_list,
        "rendered_not_kept": rendered_not_kept_list,
        "kept_and_laid": kept_and_laid_list,
        "original_while_tracked": list1,
        "seams": list2,
        "emptied_inside_a_piece": list3,
        "original_tracking_not_known": no_capture_list,
        "original_her_track_empty": empty,
        "carried_not_known": carried_unknown,
        "left_out_by_a_look": looked_out,
        "laid_but_barely_changed": barely,
        "region_far_below_its_own_size": small,
        "read_for_looks": {"ledger": str(ledger_path) if ledger_path else None, "shots": str(shots_path) if shots_path else None},
        "clip_check": {"file": str(check_path), "sha256": hashlib.sha256(check_path.read_bytes()).hexdigest()},
        "runs": runs_out,
        "by_shot": by_shot(runs_out),
    }


#: The kinds of run counted per shot (`by_shot`), in two lists the lead ruled apart (2026-10-16): what a remake/render/lay answers
#: (a laid piece failing, or captured/rendered and not kept), and what a capture answers (`no_capture`).
ORIGINAL_SHOWING = {
    "remake": REMAKE_GAP_KINDS + ("original_while_tracked",),
    "capture": CAPTURE_GAP_KINDS + ("original_tracking_not_known",),
}


def by_shot(runs: list[dict]) -> dict[str, list[dict]]:
    """Per shot, the frames of each kind where the original shows (`ORIGINAL_SHOWING`), worst first, in its two lists: which shots
    to remake first, and which want a capture. A frame two subjects show the original on is counted for each."""
    return {which: ranked(runs, kinds) for which, kinds in ORIGINAL_SHOWING.items()}


def ranked(runs: list[dict], kinds: tuple) -> list[dict]:
    shots: dict[tuple, dict] = {}
    for r in runs:
        if r["kind"] in kinds:
            x = shots.setdefault(tuple(r["shot"]), {"shot": list(r["shot"]), "frames": 0, "by_kind": {}, "subjects": set()})
            n = r["last"] - r["first"] + 1
            x["frames"] += n
            x["by_kind"][r["kind"]] = x["by_kind"].get(r["kind"], 0) + n
            x["subjects"].add(r["subject"])
    return [{**x, "subjects": sorted(x["subjects"])} for x in sorted(shots.values(), key=lambda x: (-x["frames"], x["shot"]))]


def _stamp_words(r: dict) -> dict:
    r["words"] = run_words(r)
    r["chip_words"] = chip_words_of(r)
    return r


def flat(list1: list[dict], list2: list[dict], no_cap: list[dict], empty: list[dict], written: dict[str, str],
         carried_unknown: list[dict] | None = None, looked_out: list[dict] | None = None, barely: list[dict] | None = None,
         small: list[dict] | None = None) -> list[dict]:
    """One list of runs for a page and the stacked view, by frame. Each run carries its `step` (1..4), `kind`, `files`
    (the files for the furthest step reached), `words` (the check's own words for the run), and `chip_words`."""
    out = []
    for x in list1:
        inside = x["piece"] is not None
        kind = x.get("kind") or ("emptied_inside" if inside else "captured_not_rendered")
        if inside:
            cause = x["state"].split("(", 1)[1].rstrip(")")
        elif kind == "rendered_not_kept":
            cause = "; ".join(_format_render_summary(u) for u in (x.get("renders") or []))
        else:
            cause = ", ".join(x.get("mask_files") or x.get("track_from") or []) or "no piece laid"
        out.append(_stamp_words({
            "first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"],
            "step": x.get("step", 4 if inside else 2),
            "kind": kind,
            "pieces": [x["piece"]] if inside else [], "render_days": [day(written.get(x["piece"]))] if inside else [],
            "cause": cause, "files": x.get("files") or {},
            "track_px_median": x["track_px"]["median"],
            "shot_has_a_piece_of_her": x["shot_has_a_piece_of_her"], "present": x["present"],
        }))
    for x in list2:
        out.append(_stamp_words({
            "first": x["at"][0], "last": x["at"][1], "shot": x["shot"], "subject": x["subject"],
            "step": 4, "kind": "seam",
            "pieces": [x["before"]["piece"], x["after"]["piece"]], "render_days": [x["before"]["day"], x["after"]["day"]],
            "cause": None,
            "files": {"pieces": [x["before"]["piece"], x["after"]["piece"]],
                      "piece_files": [f for f in (x["before"].get("file"), x["after"].get("file")) if f]},
            "track_px_median": None,
        }))
    for x in no_cap:
        out.append(_stamp_words({
            "first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"],
            "step": 1, "kind": "no_capture", "pieces": [x["piece"]] if x["piece"] else [], "render_days": [],
            "cause": x.get("why") or x.get("tracking"), "files": x.get("files") or {},
            "track_px_median": None, "present": x["present"],
        }))
    for x in empty:
        out.append(_stamp_words({
            "first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"],
            "step": x.get("step", 4 if x["piece"] else 2), "kind": "original_her_track_empty",
            "pieces": [x["piece"]] if x["piece"] else [], "render_days": [],
            "cause": x["state"], "files": x.get("files") or {}, "track_px_median": 0, "present": x["present"],
        }))
    for x in looked_out or []:
        out.append(_stamp_words({
            "first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"],
            "step": x.get("step", 2), "kind": "left_out_by_a_look",
            "pieces": [], "render_days": [], "cause": x["why"], "who": x["who"], "from": x["from"],
            "files": x.get("files") or {}, "track_px_median": None, "present": x["present"],
        }))
    for x in carried_unknown or []:
        out.append(_stamp_words({
            "first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"],
            "step": 4, "kind": "carried_not_known", "pieces": [x["piece"]] if x["piece"] else [], "render_days": [],
            "cause": x["state"], "files": x.get("files") or {}, "track_px_median": x["track_px"]["median"], "present": x["present"],
        }))
    for x in barely or []:
        sh = x["changed_share"]
        out.append(_stamp_words({
            "first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"],
            "step": 4, "kind": "laid_but_barely_changed",
            "pieces": [x["piece"]], "render_days": [day(written.get(x["piece"]))],
            "cause": f"{sh['least']:.3f} to {sh['most']:.3f} of its carried region changed (under {BARELY_CHANGED})",
            "files": x.get("files") or {}, "changed_share": sh, "track_px_median": None, "present": x["present"],
        }))
    for x in small or []:
        out.append(_stamp_words({
            "first": x["frames"][0], "last": x["frames"][1], "shot": x["shot"], "subject": x["subject"],
            "step": 4, "kind": "region_far_below_its_own_size",
            "pieces": [x["piece"]], "render_days": [day(written.get(x["piece"]))],
            "cause": f"carried {', '.join(str(p) for p in x['carried_px'])} px against its median {x['its_median_px']} "
                     f"(under {FAR_BELOW_ITS_OWN_SIZE} of it)",
            "files": x.get("files") or {}, "carried_px": x["carried_px"], "its_median_px": x["its_median_px"],
            "track_px_median": None, "present": x["present"],
        }))
    return sorted(out, key=lambda r: (r["first"], r["subject"], r["kind"]))


def counts(out: dict) -> dict:
    by_cause: dict[str, int] = {}
    for x in out["emptied_inside_a_piece"]:
        c = x["state"].split("(", 1)[1].rstrip(")")
        by_cause[c] = by_cause.get(c, 0) + x["count"]
    untracked = sum(x["count"] for x in out["original_her_track_empty"])
    untracked_tied = sum(x["count"] for x in out["original_her_track_empty"] if x.get("tied", True))
    untracked_untied = sum(x["count"] for x in out["original_her_track_empty"] if not x.get("tied", True))
    cnr_tied = sum(x["count"] for x in out["captured_not_rendered"] if x.get("tied", True))
    cnr_untied = sum(x["count"] for x in out["captured_not_rendered"] if not x.get("tied", True))
    rnk_tied_tracked = sum(x.get("tied_tracked_frames", 0) for x in out["rendered_not_kept"])
    rnk_untied = sum(x.get("untied_frames", 0) for x in out["rendered_not_kept"])
    rnk_tied_empty = sum(x.get("tied_empty_track_frames", 0) for x in out["rendered_not_kept"])
    emptied_frames = sum(x["count"] for x in out["emptied_inside_a_piece"])
    no_cap_frames = sum(x["count"] for x in out["no_capture"])
    nopiece = [x for x in out["original_while_tracked"] if x["piece"] is None]
    return {
        "no_capture": {"runs": len(out["no_capture"]), "frames": no_cap_frames},
        "captured_not_rendered": {
            "runs": len(out["captured_not_rendered"]),
            "frames": sum(x["count"] for x in out["captured_not_rendered"]),
            "tied_frames": cnr_tied,
            "untied_frames": cnr_untied,
        },
        "rendered_not_kept": {
            "runs": len(out["rendered_not_kept"]),
            "frames": sum(x["count"] for x in out["rendered_not_kept"]),
            "tied_tracked_frames": rnk_tied_tracked,
            "untied_frames": rnk_untied,
            "tied_empty_track_frames": rnk_tied_empty,
        },
        "kept_and_laid": {
            "runs": len(out["kept_and_laid"]),
            "frames": sum(x["count"] for x in out["kept_and_laid"]),
            "carried_frames": sum(x["count"] for x in out["kept_and_laid"] if x["state"].startswith("carried")),
            "region_far_below_its_own_size_frames": sum(x["count"] for x in out["kept_and_laid"] if x["state"].startswith("region far below")),
            "laid_but_barely_changed_frames": sum(x["count"] for x in out["kept_and_laid"] if x["state"] == "laid but barely changed"),
            "emptied_inside_a_piece_frames": sum(x["count"] for x in out["kept_and_laid"] if x["state"].startswith("original: emptied")),
        },
        "original_while_tracked": {
            "runs": len(out["original_while_tracked"]),
            "frames": sum(x["count"] for x in out["original_while_tracked"]),
            "tied_tracked_frames": cnr_tied + rnk_tied_tracked + emptied_frames,
        },
        "of_which_no_piece_laid": {
            "in_a_shot_with_a_piece_of_her": sum(x["count"] for x in nopiece if x["shot_has_a_piece_of_her"]),
            "in_a_shot_with_no_piece_of_her": sum(x["count"] for x in nopiece if not x["shot_has_a_piece_of_her"]),
        },
        "seams": len(out["seams"]),
        "emptied_inside_a_piece": {
            "runs": len(out["emptied_inside_a_piece"]),
            "frames": emptied_frames,
            "frames_by_cause": by_cause,
        },
        "original_tracking_not_known_frames": no_cap_frames,
        "original_tracking_not_in_tied_captures_frames": no_cap_frames + cnr_untied + rnk_untied + untracked_untied,
        "original_and_her_track_empty_frames": untracked,
        "original_and_her_track_empty_in_tied_captures_frames": untracked_tied + rnk_tied_empty,
        "left_out_by_a_look_frames": sum(x["count"] for x in out.get("left_out_by_a_look", [])),
        "laid_but_barely_changed": {"runs": len(out["laid_but_barely_changed"]), "frames": sum(x["count"] for x in out["laid_but_barely_changed"])},
        "region_far_below_its_own_size": {
            "runs": len(out["region_far_below_its_own_size"]),
            "frames": sum(x["count"] for x in out["region_far_below_its_own_size"]),
        },
    }


def fr(x) -> str:
    return f"{x[0]}-{x[1]}" if x[0] != x[1] else str(x[0])


def markdown(out: dict) -> str:
    c = out["counts"]
    lines = [f"# Frame coverage: {out['table']}", "", out["made"], "", f"Command: `{out['command']}`", "",
             f"Measured on: {out['measured_on']}", "", f"Cannot see: {out['cannot_see']}", ""]
    if out["failures"]:
        lines += ["## FAILED: rows whose run= does not resolve, or a file missing", ""] + [f"- {x}" for x in out["failures"]] + [""]
    for which, title in (("remake", "the shots to remake: captured or rendered and not kept, or a laid piece fails"),
                         ("capture", "the shots that want a capture: no capture under data/ holds her track")):
        kinds = REMAKE_GAP_KINDS if which == "remake" else CAPTURE_GAP_KINDS
        lines += [f"## By shot, worst first: {title}", "",
                  "| shot | frames | " + " | ".join(kinds) + " | subjects |", "|---|---|" + "---|" * len(kinds) + "---|"]
        for x in out["by_shot"][which]:
            lines.append(f"| {fr(x['shot'])} | {x['frames']} | " + " | ".join(str(x["by_kind"].get(k, 0)) for k in kinds) + f" | {', '.join(x['subjects'])} |")
        lines += [""]
    lines += ["## Counts", "", "```", json.dumps(c, indent=1), "```", "",
              "## Step 1. Nothing captured (`no_capture`)", "",
              "| shot | subject | frames | count | why | captures covering without her track |",
              "|---|---|---|---|---|---|"]
    for x in out["no_capture"]:
        without = ", ".join((x.get("files") or {}).get("captures_without_track") or []) or "-"
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['why']} | {without} |")
    lines += ["", "## Step 2. Captured, not rendered (`captured_not_rendered`)", "",
              "| shot | subject | frames | count | tied | track px least / median / most | mask files |",
              "|---|---|---|---|---|---|---|"]
    for x in out["captured_not_rendered"]:
        t = x["track_px"]
        mfiles = ", ".join(x.get("mask_files") or x.get("track_from") or [])
        lines.append(f"| {fr(x['shot'])} | {x['subject']}{'' if x['present'] == 'yes' else ' (' + str(x['present']) + ')'} | {fr(x['frames'])} | "
                     f"{x['count']} | {'yes' if x.get('tied') else 'no'} | {t['least']} / {t['median']} / {t['most']} | {mfiles} |")
    lines += ["", "## Step 3. Rendered, not kept (`rendered_not_kept`)", "",
              "| shot | subject | frames | count | clips | windows | build records |",
              "|---|---|---|---|---|---|---|"]
    for x in out["rendered_not_kept"]:
        clips_s = ", ".join(x.get("clips") or [])
        wins_s = ", ".join(x.get("windows") or []) or "-"
        builts_s = ", ".join(x.get("built_files") or []) or "-"
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {clips_s} | {wins_s} | {builts_s} |")
    lines += ["", "## Step 4. Kept and laid: emptied inside a piece (`emptied_inside`), by cause", "",
              "| shot | subject | frames | count | piece | cause | files |", "|---|---|---|---|---|---|---|"]
    for x in out["emptied_inside_a_piece"]:
        fls = x.get("files") or {}
        f_s = ", ".join(str(fls[k]) for k in ("piece_file", "region_file", "mask_file") if fls.get(k))
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['piece']} | {x['state'].split('(', 1)[1].rstrip(')')} | {f_s} |")
    lines += ["", "## Seams: two laid renders of one person meeting", "", "| shot | subject | at | before (rendered) | after (rendered) |", "|---|---|---|---|---|"]
    for x in out["seams"]:
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {x['at'][0]}/{x['at'][1]} | {x['before']['piece']} ({x['before']['day']}) | "
                     f"{x['after']['piece']} ({x['after']['day']}) |")
    lines += ["", "## Left out by a look (the shots record): the original, on a person's word", "",
              "| shot | subject | frames | count | who | why |", "|---|---|---|---|---|---|"]
    for x in out.get("left_out_by_a_look", []):
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['who']} | {str(x['why'])[:300]} |")
    lines += ["", "## She is tracked and whether the laid piece carried a mask is not known", "",
              "| shot | subject | frames | count | piece | why | track px median |", "|---|---|---|---|---|---|---|"]
    for x in out["carried_not_known"]:
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['piece'] or '-'} | {x['state']} | {x['track_px']['median']} |")
    lines += ["", f"## Laid but barely changed: a piece carried a mask and changed under {BARELY_CHANGED} of its region", "",
              f"Read from the clip's check `{out['clip_check']['file']}` (regions.pieces.<piece>.changed_px_per_frame, the assembler's `changed`), "
              "against each run's carried region.", "",
              "| shot | subject | frames | count | piece | changed share least / median / most |", "|---|---|---|---|---|---|"]
    for x in out["laid_but_barely_changed"]:
        sh = x["changed_share"]
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['piece']} | {sh['least']} / {sh['median']} / {sh['most']} |")
    lines += ["", f"## Region far below its own size: a carried region under {FAR_BELOW_ITS_OWN_SIZE} of the piece's own median", "",
              "| shot | subject | frames | count | piece | carried px | its median px |", "|---|---|---|---|---|---|---|"]
    for x in out["region_far_below_its_own_size"]:
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['piece']} | "
                     f"{', '.join(str(p) for p in x['carried_px'])} | {x['its_median_px']} |")
    lines += ["", "## The original shows on frames her track is 0 px", "", "| shot | subject | frames | count | piece | state |", "|---|---|---|---|---|---|"]
    for x in out["original_her_track_empty"]:
        lines.append(f"| {fr(x['shot'])} | {x['subject']} | {fr(x['frames'])} | {x['count']} | {x['piece'] or '(none laid)'} | {x['state']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    ap.add_argument("--table", required=True, help="a delivery table's .json (bench/delivery_table.py)")
    ap.add_argument("--span", help="A-B: only the shots that meet these frames, cut to them")
    ap.add_argument("--data", help="the data/ directory holding captures and *_job build records")
    ap.add_argument("--ledger", help="the render ledger whose records name each laid piece's job file; the one the table names when left out")
    ap.add_argument("--shots", help="the shots record whose looks are read; the one the table names when left out")
    ap.add_argument("--check", help="the assembled clip's <clip>.mp4.check.json; the one the table's assembly command writes when left out")
    ap.add_argument("--json", help="where the JSON goes; beside the table when left out")
    ap.add_argument("--md", help="where the .md goes; beside the table when left out")
    a = ap.parse_args()
    table_path = Path(a.table)
    table = json.loads(table_path.read_text())
    span = [int(x) for x in a.span.split("-")] if a.span else None
    try:
        out = coverage(table, span, data=Path(a.data) if a.data else None,
                       ledger=Path(a.ledger) if a.ledger else None, shots=Path(a.shots) if a.shots else None,
                       check=Path(a.check) if a.check else None)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 2
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stem = "frame_coverage_" + table_path.stem.removeprefix("t_") + (f"_{span[0]:04d}_{span[1]:04d}" if span else "")
    out = {"table": str(table_path), "table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(), "span": span,
           "made": f"Made {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} at commit {commit}",
           "command": " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv]),
           "measured_on": "the table's reading (shots and subjects) and rows (laid pieces, run=); every capture under data/ (tied and untied) "
                          "for her SAM 3.1 track (subjects/<label>/masks__<by>.npz); build records (*_job/*.render.built.json and *.render.queued.json) "
                          "for rendered clips not in the table and their windows; each laid run's carried mask (runs/<run>/region.npz), "
                          "its manifest's held, drop=, grade and no-mask lists and part_grades.json; the clip's check (each piece's changed px a frame)",
           "cannot_see": CANNOT, **out}
    out["counts"] = counts(out)
    jp = Path(a.json) if a.json else table_path.parent / f"{stem}.json"
    mp = Path(a.md) if a.md else table_path.parent / f"{stem}.md"
    jp.write_text(json.dumps(out, indent=None, separators=(",", ":")) + "\n")
    mp.write_text(markdown(out))
    print(json.dumps(out["counts"], indent=1))
    for which, title in (("remake", "TO REMAKE (captured or rendered and not kept, or a laid piece fails)"),
                         ("capture", "TO CAPTURE (no capture under data/ holds her track)")):
        print(f"BY SHOT, worst first, {title}:")
        for x in out["by_shot"][which][:15]:
            print(f"  {fr(x['shot'])}: {x['frames']} frame(s): " + ", ".join(f"{k} {v}" for k, v in x["by_kind"].items()) + f" ({', '.join(x['subjects'])})")
    print(f"wrote {jp}\nwrote {mp}")
    for x in out["failures"]:
        print("FAIL", x)
    return 1 if out["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
