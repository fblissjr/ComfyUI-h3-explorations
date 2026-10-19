#!/usr/bin/env python3
"""Rebuild the delivery ledger, table, per-frame gap check, optional joined clip and stack, review pages, and close-of-day answers snapshot.

    <python> bench/rebuild_delivery.py --video <stem> [--clip] [--share <dir>] [--inputs <dir>]

Replaces the untracked per-session `rebuild.sh` and `pages.sh` shell scripts with one tracked tool (`docs/wiki/runbook.md`,
sections `(c)` and `(d)`). Runs the provenance and delivery chain in strict order and stops at the first refusal, naming the
step that failed and keeping each step's `.out` log under `data/<day>_<video>_delivery_tables/`:

1. `laid_verify`: `bench/laid.py verify` checks that every entry in `data/<video>/laid.jsonl` exists on the share and matches its `sha256`.
2. `ledger`: `bench/render_ledger.py` walks the share and `data/` build records to write `ledger_<video>_<stamp>.json`, `.md`, and `_summary.md`.
3. `delivery_table`: `bench/delivery_table.py` builds `t_<video>_<first>_<last>_<stamp>.txt`, `.json`, and `reading_<video>_<first>_<last>_<stamp>.md` from `--laid`.
4. `assemble` (when `--clip` is given): `bench/laid.py verify`, then `bench/assemble_delivery.py` from the table's own `assembly.command`, then `bench/laid.py verify`, recording `assembler_blob_sha` in `.check.json`.
5. `gap_check`: when running without `--clip`, `verify_reused_clip` refuses the reused clip unless its `.check.json` records the current `assembler_blob_sha`, its table rows and each piece's `laid_by` rule match what the assembler would lay now, and its sidecar `.json` records the same `laid.jsonl` and `.check.json` `sha256`; then `bench/frame_coverage.py` writes `frame_coverage_<video>_<first>_<last>_<stamp>.json` and `.md` against the table and `.check.json`.
6. `stack` (when `--clip` is given and not `--no-stack`): `bench/stack_delivery.py build` with `--coverage` from step 5, then `bench/stack_delivery.py check`.
7. `pages` (unless `--no-pages`): `bench/covered_items.py` updates `data/<video>/review/covered.json`, then `bench/review_pages.py to-review` and `provenance` write `TO_REVIEW.html` and `PROVENANCE.html` in `Video/<video>/` on the share.
8. `snapshot` (unless `--no-snapshot`): copies `data/<video>/owner_answers.jsonl` to `data/<video>/owner_answers_snapshots/owner_answers_<stamp>.jsonl` with mode `0444` (read-only) and verifies `sha256` equality.

Every date and stamp comes from `datetime.datetime.now(datetime.timezone.utc)` unless `--day` or `--stamp` is given; `--share`
and `--inputs` default to `H3_OUTPUT_DIR` and `H3_INPUT_DIR`; table folders, extra captures, spots files, previous/watched tables,
day piece folders, and existing clip checks are discovered by deterministic rules under `data/` and `Video/<video>/`.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent

STEPS_WITHOUT_CLIP = ("laid_verify", "ledger", "delivery_table", "gap_check", "pages", "snapshot")
STEPS_WITH_CLIP = ("laid_verify", "ledger", "delivery_table", "assemble", "gap_check", "stack", "pages", "snapshot")


class Refused(Exception):
    """Raised when a precondition or a step in the delivery rebuild chain fails."""

    def __init__(self, step: str, detail: str):
        self.step = step
        self.detail = detail
        super().__init__(f"step {step} failed: {detail}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_stamp(now: datetime.datetime | None = None, day_override: str | None = None, stamp_override: str | None = None) -> tuple[str, str]:
    """Return `(day, stamp)` in UTC (`YYYY-MM-DD`, `YYYY-MM-DDTHHMMZ`)."""
    dt = (now or datetime.datetime.now(datetime.timezone.utc)).astimezone(datetime.timezone.utc)
    day = day_override or dt.strftime("%Y-%m-%d")
    stamp = stamp_override or f"{day}T{dt.strftime('%H%M')}Z"
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise Refused("init", f"--day {day!r} is not YYYY-MM-DD")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{4}Z", stamp):
        raise Refused("init", f"--stamp {stamp!r} is not YYYY-MM-DDTHHMMZ")
    return day, stamp


def resolve_share(share_arg: str | None = None) -> Path:
    """Resolve the server's output root from `--share` or `H3_OUTPUT_DIR`."""
    raw = share_arg or os.environ.get("H3_OUTPUT_DIR")
    if not raw:
        raise Refused("init", "give --share or set H3_OUTPUT_DIR to the server's output root")
    p = Path(raw).resolve()
    if not p.is_dir():
        raise Refused("init", f"share folder is not a directory: {p}")
    return p


def resolve_inputs(inputs_arg: str | None = None, repo: Path = REPO) -> Path:
    """Resolve the server's input root from `--inputs`, `H3_INPUT_DIR`, or the ComfyUI checkout's `input/`."""
    raw = inputs_arg or os.environ.get("H3_INPUT_DIR")
    if raw:
        p = Path(raw).resolve()
    else:
        p = (repo.parents[1] / "input").resolve()
    if not p.is_dir():
        raise Refused("init", f"inputs folder is not a directory: {p} (give --inputs or set H3_INPUT_DIR)")
    return p


def source_info(data_dir: Path, video: str, inputs_dir: Path, source_arg: str | None = None, span_arg: str | None = None) -> tuple[Path, int, int]:
    """Resolve `(source_video_path, first_frame, last_frame)` from `data/<video>/manifest.json` and `shots.json`."""
    vdir = data_dir / video
    shots_path = vdir / "shots.json"
    if not shots_path.is_file():
        raise Refused("init", f"shots file is not there: {shots_path}")
    shots_data = json.loads(shots_path.read_text())
    shots_list = shots_data.get("shots") or []
    if not shots_list:
        raise Refused("init", f"shots file {shots_path} lists no shots")

    manifest_path = vdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    derived = manifest.get("derived") or {}
    src_block = manifest.get("source") or {}

    if span_arg:
        m = re.fullmatch(r"(\d+)-(\d+)", span_arg.strip())
        if not m:
            raise Refused("init", f"--span {span_arg!r} must be FIRST-LAST in frame numbers")
        s0, s1 = int(m.group(1)), int(m.group(2))
        if s1 < s0:
            raise Refused("init", f"--span {span_arg!r} has last < first")
    else:
        s0 = int(shots_list[0]["first"])
        frames = derived.get("frames") or src_block.get("frames")
        s1 = (int(frames) - 1) if frames is not None else int(shots_list[-1]["last"])

    if source_arg:
        src_path = Path(source_arg).resolve()
        if not src_path.is_file():
            raise Refused("init", f"--source file is not there: {src_path}")
        return src_path, s0, s1

    file_name = (
        derived.get("name")
        or derived.get("file")
        or src_block.get("name")
        or src_block.get("file")
        or shots_data.get("video")
        or shots_data.get("clip")
        or f"{video}.mp4"
    )
    direct = inputs_dir / str(file_name)
    if direct.is_file():
        return direct.resolve(), s0, s1
    matches = sorted(p.resolve() for p in inputs_dir.rglob(Path(str(file_name)).name) if p.is_file())
    if len(matches) == 1:
        return matches[0], s0, s1
    if len(matches) > 1:
        raise Refused("init", f"more than one file named {file_name!r} under {inputs_dir}; give --source")
    raise Refused("init", f"source video {file_name!r} not found under {inputs_dir}; give --source")


def discover_table_dirs(data_dir: Path, video: str, out_dir: Path | None = None) -> list[Path]:
    """Find every `data/*_<video>_delivery_tables` folder and any immediate subfolder holding `t_*.json`."""
    bases = sorted(p for p in data_dir.glob(f"*_{video}_delivery_tables") if p.is_dir())
    if out_dir is not None and out_dir not in bases and out_dir.is_dir():
        bases.append(out_dir)
        bases.sort()
    found: list[Path] = []
    seen: set[Path] = set()
    for base in bases:
        for cand in [base, *sorted(p for p in base.iterdir() if p.is_dir() and any(p.glob("t_*.json")))]:
            rc = cand.resolve()
            if rc not in seen:
                seen.add(rc)
                found.append(cand)
    return found


def discover_also_captures(data_dir: Path, video: str) -> list[Path]:
    """Find preview/span captures `data/*_<video>_*_capture` with a `manifest.json`, keeping the newest per span group."""
    groups: dict[tuple[str, str], tuple[tuple[str, int, str], Path]] = {}
    pat = re.compile(rf"^(?P<date>\d{{4}}-\d{{2}}-\d{{2}})_{re.escape(video)}_(?P<mid>.+)_(?P<len>\d+_capture)$")
    for d in sorted(p for p in data_dir.glob(f"*_{video}_*_capture") if p.is_dir() and (p / "manifest.json").is_file()):
        m = pat.match(d.name)
        if m:
            mid = m.group("mid")
            stem = re.sub(r"\d+$", "", mid)
            rev_str = mid[len(stem):]
            rev = int(rev_str) if rev_str else 1
            gkey = (stem, m.group("len"))
            skey = (m.group("date"), rev, d.name)
        else:
            gkey = (d.name, "")
            skey = ("", 1, d.name)
        prev = groups.get(gkey)
        if prev is None or skey >= prev[0]:
            groups[gkey] = (skey, d)
    return [groups[k][1] for k in sorted(groups)]


def discover_spots(data_dir: Path, video: str) -> list[Path]:
    """Find every `data/<video>/joined_clip_spots_*.json` in sorted order."""
    vdir = data_dir / video
    if not vdir.is_dir():
        return []
    return sorted(p for p in vdir.glob("joined_clip_spots_*.json") if p.is_file())


def discover_whole_tables(data_dir: Path, video: str, s0: int, s1: int, exclude_stamp: str | None = None) -> list[Path]:
    """Find whole-span table JSONs `t_<video>_<s0:04d>_<s1:04d>_<stamp>.json` across `data/*_<video>_delivery_tables/`, sorted by stamp."""
    prefix = f"t_{video}_{s0:04d}_{s1:04d}_"
    candidates: list[tuple[str, Path]] = []
    for base in sorted(p for p in data_dir.glob(f"*_{video}_delivery_tables") if p.is_dir()):
        for f in sorted(base.glob(f"{prefix}*.json")):
            if not f.is_file():
                continue
            stamp = f.stem.removeprefix(prefix)
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{4}Z", stamp):
                continue
            if exclude_stamp and stamp == exclude_stamp:
                continue
            candidates.append((stamp, f))
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    return [p for _, p in candidates]


def discover_previous_table(data_dir: Path, video: str, s0: int, s1: int, exclude_stamp: str | None = None) -> Path | None:
    """Return the newest whole-span table JSON before `exclude_stamp`, or None."""
    tables = discover_whole_tables(data_dir, video, s0, s1, exclude_stamp=exclude_stamp)
    return tables[-1] if tables else None


def discover_watched_table(data_dir: Path, video: str, s0: int, s1: int, exclude_stamp: str | None = None) -> Path | None:
    """Return the earliest whole-span table JSON before `exclude_stamp`, or None."""
    tables = discover_whole_tables(data_dir, video, s0, s1, exclude_stamp=exclude_stamp)
    return tables[0] if tables else None


def discover_day_folder(share_dir: Path, video: str, day: str) -> Path:
    """Return the newest `Video/<video>/day_*` folder holding `[0-9][0-9][0-9][0-9]_*` piece folders, or `Video/<video>/day_<day>`."""
    vdir = share_dir / "Video" / video
    if vdir.is_dir():
        with_pieces = sorted(
            d for d in vdir.glob("day_*")
            if d.is_dir() and any(p.is_dir() for p in d.glob("[0-9][0-9][0-9][0-9]_*"))
        )
        if with_pieces:
            return with_pieces[-1]
        all_days = sorted(d for d in vdir.glob("day_*") if d.is_dir())
        if all_days:
            return all_days[-1]
    return vdir / f"day_{day}"


def discover_existing_clip(share_dir: Path, video: str, s0: int, s1: int) -> Path | None:
    """Find the newest assembled joined clip `Video/<video>/day_*/<video>_<s0:04d>_<s1:04d>_joined_*.mp4` that has a `.mp4.check.json`."""
    vdir = share_dir / "Video" / video
    if not vdir.is_dir():
        return None
    pattern = f"day_*/{video}_{s0:04d}_{s1:04d}_joined_*.mp4"
    found: list[tuple[str, str, Path]] = []
    for p in sorted(vdir.glob(pattern)):
        if p.name.endswith("_STACKED.mp4") or ".1fps." in p.name or not p.is_file():
            continue
        check_path = Path(str(p) + ".check.json")
        if check_path.is_file():
            found.append((p.parent.name, p.name, p))
    if not found:
        return None
    found.sort(key=lambda item: (item[0], item[1]))
    return found[-1][2]


def assembler_blob_sha(repo: Path = REPO) -> str:
    """Return the git blob SHA-1 of `bench/assemble_delivery.py`."""
    raw = (repo / "bench" / "assemble_delivery.py").read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def stamp_assembler_blob_sha(clip_out: Path, check_json: Path, repo: Path = REPO) -> None:
    """Record `assembler_blob_sha` in `check_json` and update its digest in `<clip_out>.json` if needed."""
    if not check_json.is_file():
        return
    try:
        chk_doc = json.loads(check_json.read_text())
    except ValueError:
        return
    if not isinstance(chk_doc, dict) or not chk_doc:
        return
    blob_sha = assembler_blob_sha(repo)
    if chk_doc.get("assembler_blob_sha") != blob_sha:
        chk_doc["assembler_blob_sha"] = blob_sha
        check_json.write_text(json.dumps(chk_doc, indent=1) + "\n")
        sidecar_json = clip_out.with_name(clip_out.name + ".json")
        if sidecar_json.is_file():
            try:
                sc_doc = json.loads(sidecar_json.read_text())
                for item in sc_doc.get("inputs") or []:
                    if isinstance(item, dict) and item.get("role") == "the check record":
                        item["sha256"] = sha256_file(check_json)
                        item["size"] = check_json.stat().st_size
                sidecar_json.write_text(json.dumps(sc_doc, indent=2) + "\n")
            except ValueError:
                pass


def _now_laid_by(piece_path: str, tr: dict, check_data: dict, repo: Path) -> dict:
    """Return `assemble_delivery.laid_by_of(piece_path, run)` as the current assembler would lay `piece_path`."""
    import assemble_delivery as ad

    run_tuple = None
    run_spec = tr.get("run")
    if isinstance(run_spec, str) and ":" in run_spec:
        where, _, rname = run_spec.partition(":")
        cands = [Path(where), repo / where]
        cands.extend(Path(c) for c in (check_data.get("captures") or []) if Path(c).name == Path(where).name)
        for cand in cands:
            if (cand / "manifest.json").is_file():
                try:
                    m = json.loads((cand / "manifest.json").read_text())
                    rentry = next((x for x in (m.get("runs") or []) if x.get("name") == rname), None)
                    if rentry is not None:
                        run_tuple = (cand, m, rentry)
                        break
                except ValueError:
                    pass
    if run_tuple is None:
        for cap_str in check_data.get("captures") or []:
            cand = Path(cap_str)
            if (cand / "manifest.json").is_file():
                try:
                    m = json.loads((cand / "manifest.json").read_text())
                    rentry = next(
                        (x for x in (m.get("runs") or []) if x.get("render") and Path(str(x["render"])).name == Path(piece_path).name),
                        None,
                    )
                    if rentry is not None:
                        run_tuple = (cand, m, rentry)
                        break
                except ValueError:
                    pass
    return ad.laid_by_of(piece_path, run_tuple)


def verify_reused_clip(clip_out: Path, check_json: Path, T: dict, t_json: Path, laid_path: Path, share: Path, repo: Path = REPO) -> None:
    """Refuse reusing an existing clip unless its `.check.json` and sidecar match the current assembler, `laid.jsonl`, table rows, and `laid_by` rules."""
    if not check_json.is_file():
        raise Refused("gap_check", f"clip check is not there: {check_json}; pass --clip to assemble first")
    try:
        check_data = json.loads(check_json.read_text())
    except ValueError as e:
        raise Refused("gap_check", f"clip check {check_json.name} is not valid JSON: {e}") from None

    want_blob = assembler_blob_sha(repo)
    got_blob = check_data.get("assembler_blob_sha")
    if not got_blob:
        raise Refused(
            "gap_check",
            f"reused clip {clip_out.name} check record {check_json.name} does not record assembler_blob_sha; pass --clip to reassemble",
        )
    if got_blob != want_blob:
        raise Refused(
            "gap_check",
            f"reused clip {clip_out.name} was built with assemble_delivery.py blob {str(got_blob)[:12]}, "
            f"not current {want_blob[:12]}; pass --clip to reassemble",
        )

    def _norm_row(r: dict, is_check: bool) -> tuple:
        if is_check:
            fr = r.get("frames") or [r.get("first"), r.get("last")]
            raw_p = str(r.get("piece") or r.get("file") or "")
        else:
            fr = [r.get("first"), r.get("last")]
            raw_p = str(r.get("file") or r.get("piece") or "")
        p_obj = Path(raw_p)
        resolved = str((p_obj if p_obj.is_absolute() else (share / p_obj)).resolve())
        return (int(fr[0]), int(fr[1]), resolved, int(r.get("piece_first", fr[0])), list(r.get("restore") or []))

    raw_check_rows = check_data.get("table")
    if not isinstance(raw_check_rows, list):
        raise Refused("gap_check", f"reused clip {clip_out.name} check record {check_json.name} has no table row list; pass --clip to reassemble")
    check_rows = [_norm_row(r, is_check=True) for r in raw_check_rows]
    table_rows = [_norm_row(r, is_check=False) for r in (T.get("rows") or [])]
    if check_rows != table_rows:
        raise Refused(
            "gap_check",
            f"reused clip {clip_out.name} check record {check_json.name} table rows do not match {t_json.name}; pass --clip to reassemble",
        )

    pieces_reg = (check_data.get("regions") or {}).get("pieces") or {}
    for tr, (_, _, resolved_piece, _, _) in zip(T.get("rows") or [], table_rows):
        rec_piece = pieces_reg.get(resolved_piece)
        if rec_piece is None:
            rec_piece = next(
                (v for k, v in pieces_reg.items() if Path(str(k)).resolve() == Path(resolved_piece).resolve()),
                None,
            )
        rec_lb = (rec_piece or {}).get("laid_by") if isinstance(rec_piece, dict) else None
        if not isinstance(rec_lb, dict) or not rec_lb.get("rule"):
            raise Refused(
                "gap_check",
                f"reused clip {clip_out.name} check record {check_json.name} has no laid_by rule for {Path(resolved_piece).name}; pass --clip to reassemble",
            )
        now_lb = _now_laid_by(resolved_piece, tr, check_data, repo)
        if rec_lb.get("rule") != now_lb.get("rule") or rec_lb.get("windows_without_a_matte") != now_lb.get("windows_without_a_matte"):
            raise Refused(
                "gap_check",
                f"reused clip {clip_out.name} laid {Path(resolved_piece).name} by rule {rec_lb.get('rule')!r}, "
                f"but assembler would now lay it by {now_lb.get('rule')!r}; pass --clip to reassemble",
            )

    sidecar_json = clip_out.with_name(clip_out.name + ".json")
    if not sidecar_json.is_file():
        raise Refused("gap_check", f"reused clip {clip_out.name} has no sidecar {sidecar_json.name}; pass --clip to reassemble")
    try:
        sc = json.loads(sidecar_json.read_text())
    except ValueError as e:
        raise Refused("gap_check", f"reused clip sidecar {sidecar_json.name} is not valid JSON: {e}") from None
    inputs = sc.get("inputs") if isinstance(sc.get("inputs"), list) else []
    chk_in = next((x for x in inputs if isinstance(x, dict) and x.get("role") == "the check record"), None)
    now_chk_sha = sha256_file(check_json)
    if not chk_in or chk_in.get("sha256") != now_chk_sha:
        raise Refused(
            "gap_check",
            f"reused clip {clip_out.name} sidecar {sidecar_json.name} check record sha256 {str((chk_in or {}).get('sha256'))[:12]} "
            f"does not match {check_json.name} ({now_chk_sha[:12]}); pass --clip to reassemble",
        )
    laid_in = next((x for x in inputs if isinstance(x, dict) and x.get("role") == "the list of what is laid"), None)
    now_laid_sha = sha256_file(laid_path)
    if not laid_in or laid_in.get("sha256") != now_laid_sha:
        raise Refused(
            "gap_check",
            f"reused clip {clip_out.name} was built from {laid_path.name} sha256 {str((laid_in or {}).get('sha256'))[:12]}, "
            f"not current {laid_path.name} ({now_laid_sha[:12]}); pass --clip to reassemble",
        )


def verify_poc_on_disk(vdir: Path) -> None:
    """When `data/<video>/poc/` exists, verify on disk that `poc/index.html`, `poc/who/index.html`, and every local
    file and folder they fetch or reference exist (`C61` item 2c). No live HTTP is used."""
    poc = vdir / "poc"
    if not poc.is_dir():
        return
    for rel in ("index.html", "who/index.html"):
        p = poc / rel
        if not p.is_file():
            raise Refused("pages", f"poc page is not there on disk: poc/{rel}")
    for rel in ("index.html", "who/index.html"):
        page = poc / rel
        base = page.parent
        txt = page.read_text(errors="replace")
        fetches = re.findall(r"\b(?:fetch|get)\(\s*[\"\x27]([^\"\x27]+)[\"\x27]", txt)
        media_srcs = re.findall(r"<(?:img|video|source|script)\b[^>]*?\bsrc=\"([^\"]+)\"", txt)
        if any(u.startswith("/api/answer") for u in fetches):
            if not (poc / "answers.jsonl").is_file() and not (vdir / "owner_answers.jsonl").is_file():
                raise Refused("pages", f"poc/{rel} fetches /api/answers, but neither poc/answers.jsonl nor owner_answers.jsonl is on disk")
        for u in sorted(set(fetches + media_srcs)):
            u_clean = u.split("?", 1)[0].split("#", 1)[0].strip()
            if not u_clean or "://" in u_clean or u_clean.startswith(("/api/", "data:", "blob:")):
                continue
            target = base / u_clean
            rel_parent = target.parent.relative_to(poc) if target.parent.is_relative_to(poc) else target.parent
            rel_target = target.relative_to(poc) if target.is_relative_to(poc) else target
            if not target.parent.is_dir():
                raise Refused("pages", f"poc/{rel} references folder poc/{rel_parent}, which is not on disk")
            if not target.is_file():
                raise Refused("pages", f"poc/{rel} fetches or loads poc/{rel_target}, which is not on disk")
            if target.suffix == ".json":
                try:
                    doc = json.loads(target.read_text())
                except ValueError as e:
                    raise Refused("pages", f"poc/{rel_target} is not valid JSON: {e}") from None
                if isinstance(doc, dict):
                    pics = list(doc.get("pictures") or [])
                    for r in (doc.get("rows") or []):
                        if isinstance(r, dict):
                            pics.extend(r.get("pictures") or [])
                    for pic in pics:
                        if isinstance(pic, dict):
                            for k in ("tile", "plain"):
                                val = pic.get(k)
                                if isinstance(val, str) and val:
                                    pt = base / val
                                    rel_pt_dir = pt.parent.relative_to(poc) if pt.parent.is_relative_to(poc) else pt.parent
                                    rel_pt = pt.relative_to(poc) if pt.is_relative_to(poc) else pt
                                    if not pt.parent.is_dir():
                                        raise Refused("pages", f"poc/{rel_target} references folder poc/{rel_pt_dir}, which is not on disk")
                                    if not pt.is_file():
                                        raise Refused("pages", f"poc/{rel_target} references poc/{rel_pt}, which is not on disk")


def snapshot_owner_answers(data_dir: Path, video: str, stamp: str) -> dict:
    """Copy `data/<video>/owner_answers.jsonl` to `data/<video>/owner_answers_snapshots/owner_answers_<stamp>.jsonl` with mode `0444`."""
    src = data_dir / video / "owner_answers.jsonl"
    if not src.is_file():
        raise Refused("snapshot", f"owner answers file is not there: {src}")
    src_sha = sha256_file(src)
    snap_dir = data_dir / video / "owner_answers_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    dst = snap_dir / f"owner_answers_{stamp}.jsonl"
    if dst.exists():
        dst_sha = sha256_file(dst)
        if dst_sha != src_sha:
            raise Refused("snapshot", f"existing snapshot {dst} has sha256 {dst_sha[:12]}, which differs from {src} ({src_sha[:12]})")
        os.chmod(dst, 0o444)
    else:
        shutil.copyfile(src, dst)
        os.chmod(dst, 0o444)
    dst_sha = sha256_file(dst)
    if dst_sha != src_sha:
        raise Refused("snapshot", f"snapshot {dst} sha256 {dst_sha[:12]} does not match {src} ({src_sha[:12]})")
    mode = stat.S_IMODE(os.stat(dst).st_mode)
    if mode & 0o222:
        raise Refused("snapshot", f"snapshot {dst} is writable (mode {oct(mode)}); expected 0o444")
    return {"source": str(src), "snapshot": str(dst), "sha256": dst_sha, "mode": f"0o{mode:03o}"}


def rel_or_name(path: Path, repo: Path, share: Path) -> str:
    """Format a path relative to `repo` or `share` so no machine mount or home path enters a record."""
    rp = path.resolve()
    for base in (repo.resolve(), share.resolve()):
        try:
            return str(rp.relative_to(base))
        except ValueError:
            continue
    return path.name


def run_step(
    step: str,
    cmd: list[str],
    log_path: Path,
    cwd: Path,
    env: dict[str, str],
    runner: Callable[[str, list[str], Path, Path, dict[str, str]], tuple[int, str]] | None = None,
) -> str:
    """Run one command, write its combined output to `log_path`, and raise `Refused(step, ...)` on non-zero exit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if runner is not None:
        rc, out = runner(step, list(cmd), log_path, cwd, env)
        log_path.write_text(out)
    else:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        rc, out = proc.returncode, proc.stdout
        log_path.write_text(out)
    if rc != 0:
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        tail = lines[-1] if lines else f"exit status {rc}"
        raise Refused(step, f"exit {rc} (log: {log_path.name}): {tail}")
    return out


def replace_python_cmd(line: str, python_exe: str) -> list[str]:
    """Parse a shell command line emitted by `delivery_table.py` and run it with `python_exe`."""
    words = shlex.split(line)
    if not words:
        raise ValueError("empty command line")
    if Path(words[0]).name.startswith("python"):
        words[0] = python_exe
    return words


def rebuild(
    a: argparse.Namespace,
    repo: Path = REPO,
    runner: Callable[[str, list[str], Path, Path, dict[str, str]], tuple[int, str]] | None = None,
) -> dict:
    """Execute the delivery rebuild chain in order and return the summary record."""
    day, stamp = utc_stamp(day_override=a.day, stamp_override=a.stamp)
    share = resolve_share(a.share)
    inputs = resolve_inputs(a.inputs, repo=repo)
    data_dir = Path(a.data).resolve() if a.data else (repo / "data").resolve()
    video = a.video.strip()
    if not video or "/" in video or "\\" in video:
        raise Refused("init", f"--video {a.video!r} must be a video stem, not a path")

    vdir = data_dir / video
    if not vdir.is_dir():
        raise Refused("init", f"video data folder is not there: {vdir}")

    laid_path = Path(a.laid).resolve() if a.laid else (vdir / "laid.jsonl")
    shots_path = Path(a.shots).resolve() if a.shots else (vdir / "shots.json")
    plan_path = Path(a.plan).resolve() if a.plan else (vdir / "plan.json")
    verdicts_path = Path(a.verdicts).resolve() if a.verdicts else (vdir / "verdicts" / "verdicts.jsonl")
    answers_path = Path(a.answers).resolve() if a.answers else (vdir / "owner_answers.jsonl")
    items_path = Path(a.items).resolve() if a.items else (vdir / "review" / "items.json")
    covered_path = Path(a.covered).resolve() if a.covered else (vdir / "review" / "covered.json")

    for label, req in (("laid", laid_path), ("shots", shots_path), ("plan", plan_path), ("verdicts", verdicts_path)):
        if not req.is_file():
            raise Refused("init", f"{label} file is not there: {req}")

    src_path, s0, s1 = source_info(data_dir, video, inputs, source_arg=a.source, span_arg=a.span)
    out_dir = Path(a.out_dir).resolve() if a.out_dir else (data_dir / f"{day}_{video}_delivery_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_table_json = Path(a.previous_table).resolve() if a.previous_table else discover_previous_table(data_dir, video, s0, s1, exclude_stamp=stamp)
    prev_table_txt: Path | None = None
    if prev_table_json is not None:
        if prev_table_json.suffix == ".txt":
            prev_table_txt = prev_table_json
            prev_table_json = prev_table_json.with_suffix(".json")
        else:
            prev_table_txt = prev_table_json.with_suffix(".txt")

    watched_table_json = Path(a.watched_table).resolve() if a.watched_table else discover_watched_table(data_dir, video, s0, s1, exclude_stamp=stamp)
    table_dirs = [Path(p).resolve() for p in a.tables] if a.tables else discover_table_dirs(data_dir, video, out_dir=out_dir)
    also_captures = [Path(p).resolve() for p in a.also_capture] if a.also_capture else discover_also_captures(data_dir, video)
    spots_files = [Path(p).resolve() for p in a.spots] if a.spots else discover_spots(data_dir, video)
    day_folder = Path(a.day_folder).resolve() if a.day_folder else discover_day_folder(share, video, day)

    if a.clip_out:
        clip_out = Path(a.clip_out).resolve()
    elif a.clip:
        clip_out = (share / "Video" / video / f"day_{day}" / f"{video}_{s0:04d}_{s1:04d}_joined_{stamp}_from_laid.mp4").resolve()
    else:
        existing = discover_existing_clip(share, video, s0, s1)
        if existing is None:
            raise Refused(
                "delivery_table",
                f"no existing assembled clip with .mp4.check.json under Video/{video}/day_*/ for {s0:04d}-{s1:04d}; pass --clip or --clip-out",
            )
        clip_out = existing.resolve()

    if a.clip:
        clip_out.parent.mkdir(parents=True, exist_ok=True)

    python_exe = a.python or sys.executable
    env = dict(os.environ)
    env["H3_OUTPUT_DIR"] = str(share)
    env["H3_INPUT_DIR"] = str(inputs)

    steps_run: list[str] = []
    logs: dict[str, str] = {}

    # 1. laid_verify
    laid_log = out_dir / f"laid_verify_{video}_{stamp}.out"
    laid_cmd = [python_exe, str(BENCH / "laid.py"), "verify", "--laid", str(laid_path), "--share", str(share)]
    run_step("laid_verify", laid_cmd, laid_log, repo, env, runner=runner)
    steps_run.append("laid_verify")
    logs["laid_verify"] = rel_or_name(laid_log, repo, share)

    # 2. ledger
    ledger_json = out_dir / f"ledger_{video}_{stamp}.json"
    ledger_md = out_dir / f"ledger_{video}_{stamp}.md"
    ledger_summary = out_dir / f"ledger_{video}_{stamp}_summary.md"
    ledger_log = out_dir / f"ledger_{video}_{stamp}.out"
    ledger_cmd = [
        python_exe, str(BENCH / "render_ledger.py"),
        "--shots", str(shots_path),
        "--plan", str(plan_path),
        "--share", str(share),
        "--records", str(data_dir),
        "--copies-of", str(vdir),
        "--verdicts", str(verdicts_path),
        "--inputs", str(inputs),
        "--json", str(ledger_json),
        "--md", str(ledger_md),
        "--summary", str(ledger_summary),
    ]
    if answers_path.is_file():
        ledger_cmd += ["--answers", str(answers_path)]
    if items_path.is_file():
        ledger_cmd += ["--items", str(items_path)]
    if prev_table_txt is not None and prev_table_txt.is_file():
        ledger_cmd += ["--table", str(prev_table_txt)]
    for sp in spots_files:
        ledger_cmd += ["--spots", str(sp)]
    for td in table_dirs:
        ledger_cmd += ["--joined-tables", str(td)]
    run_step("ledger", ledger_cmd, ledger_log, repo, env, runner=runner)
    if not ledger_json.is_file() or not ledger_md.is_file():
        raise Refused("ledger", f"render_ledger.py did not write {ledger_json.name} and {ledger_md.name}")
    steps_run.append("ledger")
    logs["ledger"] = rel_or_name(ledger_log, repo, share)

    # 3. delivery_table
    span_tag = f"{s0:04d}_{s1:04d}"
    t_txt = out_dir / f"t_{video}_{span_tag}_{stamp}.txt"
    t_json = out_dir / f"t_{video}_{span_tag}_{stamp}.json"
    reading_md = out_dir / f"reading_{video}_{span_tag}_{stamp}.md"
    table_log = out_dir / f"table_{video}_{span_tag}_{stamp}.out"
    stack_work = Path(a.stack_work).resolve() if a.stack_work else (Path(tempfile.gettempdir()) / f"h3_stack_work_{video}_{stamp}")
    assemble_args = f"--source {shlex.quote(str(src_path))} --out {shlex.quote(str(clip_out))} --size {shlex.quote(a.size)}"
    table_cmd = [
        python_exe, str(BENCH / "delivery_table.py"),
        "--ledger", str(ledger_json),
        "--shots", str(shots_path),
        "--plan", str(plan_path),
        "--laid", str(laid_path),
        "--assemble-args", assemble_args,
        "--stack-work", str(stack_work),
        "--table", str(t_txt),
        "--reading", str(reading_md),
        "--json", str(t_json),
    ]
    if a.span:
        table_cmd += ["--span", f"{s0}-{s1}"]
    for cap in also_captures:
        table_cmd += ["--also-capture", str(cap)]
    run_step("delivery_table", table_cmd, table_log, repo, env, runner=runner)
    if not t_txt.is_file() or not t_json.is_file() or not reading_md.is_file():
        raise Refused("delivery_table", f"delivery_table.py did not write {t_txt.name}, {t_json.name}, and {reading_md.name}")
    steps_run.append("delivery_table")
    logs["delivery_table"] = rel_or_name(table_log, repo, share)

    T = json.loads(t_json.read_text())
    asm_block = T.get("assembly") or {}
    stack_block = asm_block.get("stack") or {}
    if not asm_block.get("whole") or not stack_block:
        raise Refused("delivery_table", f"{t_json.name} assembly block is incomplete")

    # Ensure the newly written table folder is in table_dirs for covered_items and review_pages
    if out_dir.resolve() not in {p.resolve() for p in table_dirs}:
        table_dirs.append(out_dir)

    # 4. assemble (when --clip is given)
    check_json = Path(str(clip_out) + ".check.json")
    if a.clip:
        asm_log = out_dir / f"assemble_{video}_{span_tag}_{stamp}.out"
        asm_cmd = replace_python_cmd(asm_block["command"], python_exe)
        run_step("assemble", asm_cmd, asm_log, repo, env, runner=runner)
        laid_after_log = out_dir / f"laid_after_{video}_{stamp}.out"
        run_step("assemble", laid_cmd, laid_after_log, repo, env, runner=runner)
        if not clip_out.is_file() or not check_json.is_file():
            raise Refused("assemble", f"assemble_delivery.py did not write {clip_out.name} and {check_json.name}")
        stamp_assembler_blob_sha(clip_out, check_json, repo=repo)
        steps_run.append("assemble")
        logs["assemble"] = rel_or_name(asm_log, repo, share)
    else:
        verify_reused_clip(clip_out, check_json, T, t_json, laid_path, share, repo=repo)

    # 5. gap_check (frame_coverage.py)
    cov_json = Path(stack_block["coverage_json"]).resolve() if stack_block.get("coverage_json") else (out_dir / f"frame_coverage_{video}_{span_tag}_{stamp}.json")
    cov_md = cov_json.with_suffix(".md")
    cov_log = out_dir / f"coverage_{video}_{span_tag}_{stamp}.out"
    cov_cmd = [
        python_exe, str(BENCH / "frame_coverage.py"),
        "--table", str(t_json),
        "--ledger", str(ledger_json),
        "--shots", str(shots_path),
        "--check", str(check_json),
        "--json", str(cov_json),
        "--md", str(cov_md),
    ]
    if a.span:
        cov_cmd += ["--span", f"{s0}-{s1}"]
    run_step("gap_check", cov_cmd, cov_log, repo, env, runner=runner)
    if not cov_json.is_file():
        raise Refused("gap_check", f"frame_coverage.py did not write {cov_json.name}")
    steps_run.append("gap_check")
    logs["gap_check"] = rel_or_name(cov_log, repo, share)

    # 6. stack (when --clip is given and not --no-stack)
    if a.clip and not a.no_stack:
        stack_work.mkdir(parents=True, exist_ok=True)
        stack_log = out_dir / f"stack_{video}_{span_tag}_{stamp}.out"
        stack_check_log = out_dir / f"stack_check_{video}_{span_tag}_{stamp}.out"
        stack_build_cmd = replace_python_cmd(stack_block["build"], python_exe)
        if "--data" not in stack_build_cmd:
            stack_build_cmd += ["--data", str(data_dir)]
        stack_check_cmd = replace_python_cmd(stack_block["check"], python_exe)
        run_step("stack", stack_build_cmd, stack_log, repo, env, runner=runner)
        run_step("stack", stack_check_cmd, stack_check_log, repo, env, runner=runner)
        steps_run.append("stack")
        logs["stack"] = rel_or_name(stack_log, repo, share)
        logs["stack_check"] = rel_or_name(stack_check_log, repo, share)

    # 7. pages (covered_items.py + review_pages.py to-review + provenance)
    to_review_html = share / "Video" / video / "TO_REVIEW.html"
    provenance_html = share / "Video" / video / "PROVENANCE.html"
    if not a.no_pages:
        if not items_path.is_file():
            raise Refused("pages", f"review items file is not there: {items_path}")
        to_review_html.parent.mkdir(parents=True, exist_ok=True)

        cov_items_log = out_dir / f"covered_{video}_{stamp}.out"
        cov_items_cmd = [
            python_exe, str(BENCH / "covered_items.py"),
            "--items", str(items_path),
            "--verdicts", str(verdicts_path),
            "--out", str(covered_path),
        ]
        for td in table_dirs:
            cov_items_cmd += ["--tables", str(td)]
        run_step("pages", cov_items_cmd, cov_items_log, repo, env, runner=runner)
        if not covered_path.is_file():
            raise Refused("pages", f"covered_items.py did not write {covered_path.name}")

        page_url = a.page_url or os.environ.get("H3_REVIEW_PAGE_URL") or "#"
        open_list = a.open_list or os.environ.get("H3_OPEN_LIST_URL") or "#"
        start_here_path: Path | None = Path(a.start_here).resolve() if a.start_here else None
        if start_here_path is None:
            default_sh = vdir / "review" / "start_here.json"
            if default_sh.is_file():
                start_here_path = default_sh.resolve()

        to_review_log = out_dir / f"page_to_review_{video}_{stamp}.out"
        to_review_cmd = [
            python_exe, str(BENCH / "review_pages.py"), "to-review",
            "--table", str(t_json),
            "--ledger", str(ledger_json),
            "--verdicts", str(verdicts_path),
            "--items", str(items_path),
            "--source", str(src_path),
            "--share", str(share),
            "--laid", str(laid_path),
            "--stamp", stamp,
            "--page-url", page_url,
            "--open-list", open_list,
            "--provenance", "PROVENANCE.html",
            "--out", str(to_review_html),
        ]
        if answers_path.is_file():
            to_review_cmd += ["--answers", str(answers_path)]
        if prev_table_json is not None and prev_table_json.is_file():
            to_review_cmd += ["--previous-table", str(prev_table_json)]
        if start_here_path is not None:
            to_review_cmd += ["--start-here", str(start_here_path)]
        if a.today:
            to_review_cmd += ["--today", a.today]
        for td in table_dirs:
            to_review_cmd += ["--tables", str(td)]
        run_step("pages", to_review_cmd, to_review_log, repo, env, runner=runner)

        prov_log = out_dir / f"page_provenance_{video}_{stamp}.out"
        prov_cmd = [
            python_exe, str(BENCH / "review_pages.py"), "provenance",
            "--table", str(t_json),
            "--ledger", str(ledger_json),
            "--verdicts", str(verdicts_path),
            "--items", str(items_path),
            "--source", str(src_path),
            "--share", str(share),
            "--laid", str(laid_path),
            "--stamp", stamp,
            "--record", str(check_json),
            "--day-folder", str(day_folder),
            "--up-to", str(s1),
            "--out", str(provenance_html),
        ]
        if watched_table_json is not None and watched_table_json.is_file():
            prov_cmd += ["--watched-table", str(watched_table_json)]
        for td in table_dirs:
            prov_cmd += ["--tables", str(td)]
        run_step("pages", prov_cmd, prov_log, repo, env, runner=runner)

        if not to_review_html.is_file() or not provenance_html.is_file():
            raise Refused("pages", f"review_pages.py did not write {to_review_html.name} and {provenance_html.name}")
        poc_dir = data_dir / video / "poc"
        if poc_dir.is_dir():
            shutil.copy2(to_review_html, poc_dir / "TO_REVIEW.html")
            shutil.copy2(provenance_html, poc_dir / "PROVENANCE.html")
        verify_poc_on_disk(vdir)
        steps_run.append("pages")
        logs["covered_items"] = rel_or_name(cov_items_log, repo, share)
        logs["to_review"] = rel_or_name(to_review_log, repo, share)
        logs["provenance"] = rel_or_name(prov_log, repo, share)

    # 8. snapshot (close-of-day read-only copy of owner_answers.jsonl)
    snap_info: dict | None = None
    if not a.no_snapshot:
        snap_raw = snapshot_owner_answers(data_dir, video, stamp)
        snap_log = out_dir / f"snapshot_{video}_{stamp}.out"
        snap_info = {
            "source": rel_or_name(Path(snap_raw["source"]), repo, share),
            "snapshot": rel_or_name(Path(snap_raw["snapshot"]), repo, share),
            "sha256": snap_raw["sha256"],
            "mode": snap_raw["mode"],
        }
        snap_log.write_text(
            f"snapshot {snap_info['snapshot']} from {snap_info['source']} (sha256 {snap_info['sha256']}, mode {snap_info['mode']})\n"
        )
        steps_run.append("snapshot")
        logs["snapshot"] = rel_or_name(snap_log, repo, share)

    summary = {
        "video": video,
        "day": day,
        "stamp": stamp,
        "span": [s0, s1],
        "clip_built": bool(a.clip),
        "steps": steps_run,
        "artifacts": {
            "ledger_json": rel_or_name(ledger_json, repo, share),
            "ledger_md": rel_or_name(ledger_md, repo, share),
            "ledger_summary": rel_or_name(ledger_summary, repo, share),
            "table_txt": rel_or_name(t_txt, repo, share),
            "table_json": rel_or_name(t_json, repo, share),
            "reading_md": rel_or_name(reading_md, repo, share),
            "coverage_json": rel_or_name(cov_json, repo, share),
            "coverage_md": rel_or_name(cov_md, repo, share),
            "clip": rel_or_name(clip_out, repo, share),
            "clip_check": rel_or_name(check_json, repo, share),
            "to_review_html": None if a.no_pages else rel_or_name(to_review_html, repo, share),
            "provenance_html": None if a.no_pages else rel_or_name(provenance_html, repo, share),
        },
        "discovered": {
            "previous_table": None if prev_table_json is None else rel_or_name(prev_table_json, repo, share),
            "watched_table": None if watched_table_json is None else rel_or_name(watched_table_json, repo, share),
            "day_folder": rel_or_name(day_folder, repo, share),
            "table_dirs": [rel_or_name(p, repo, share) for p in table_dirs],
            "also_captures": [rel_or_name(p, repo, share) for p in also_captures],
            "spots": [rel_or_name(p, repo, share) for p in spots_files],
        },
        "logs": logs,
        "snapshot": snap_info,
    }
    manifest_out = out_dir / f"rebuild_{video}_{stamp}.json"
    manifest_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"rebuild {video} {stamp} ({s0}-{s1}): {', '.join(steps_run)}")
    print(f"wrote {rel_or_name(manifest_out, repo, share)}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", required=True, help="the video stem under data/<video>/ and Video/<video>/")
    p.add_argument("--clip", action="store_true", help="assemble the joined delivery clip and build + check its stacked view")
    p.add_argument("--no-stack", action="store_true", help="when --clip is set, skip stack_delivery.py build and check")
    p.add_argument("--no-pages", action="store_true", help="skip covered_items.py and review_pages.py")
    p.add_argument("--no-snapshot", action="store_true", help="skip the read-only snapshot of owner_answers.jsonl")
    p.add_argument("--share", help="the server's output root (defaults to H3_OUTPUT_DIR)")
    p.add_argument("--inputs", help="the server's input root (defaults to H3_INPUT_DIR)")
    p.add_argument("--data", help="the data root (defaults to <repo>/data)")
    p.add_argument("--out-dir", help="where the ledger, table, gap check, and .out logs are written (defaults to data/<day>_<video>_delivery_tables)")
    p.add_argument("--day", help="UTC day YYYY-MM-DD (defaults to date -u)")
    p.add_argument("--stamp", help="UTC stamp YYYY-MM-DDTHHMMZ (defaults to date -u)")
    p.add_argument("--span", help="FIRST-LAST frame span (defaults to the whole video from shots.json and manifest.json)")
    p.add_argument("--size", default="source", help="--size passed to assemble_delivery.py (default: source)")
    p.add_argument("--source", help="explicit source video file path (defaults to resolving data/<video>/manifest.json under --inputs)")
    p.add_argument("--clip-out", help="explicit joined clip path (.mp4)")
    p.add_argument("--laid", help="explicit laid.jsonl path (defaults to data/<video>/laid.jsonl)")
    p.add_argument("--shots", help="explicit shots.json path (defaults to data/<video>/shots.json)")
    p.add_argument("--plan", help="explicit plan.json path (defaults to data/<video>/plan.json)")
    p.add_argument("--verdicts", help="explicit verdicts.jsonl path (defaults to data/<video>/verdicts/verdicts.jsonl)")
    p.add_argument("--answers", help="explicit owner_answers.jsonl path (defaults to data/<video>/owner_answers.jsonl)")
    p.add_argument("--items", help="explicit review items.json path (defaults to data/<video>/review/items.json)")
    p.add_argument("--covered", help="explicit covered.json path (defaults to data/<video>/review/covered.json)")
    p.add_argument("--previous-table", help="explicit previous table JSON or TXT (defaults to newest prior whole-span table)")
    p.add_argument("--watched-table", help="explicit watched table JSON (defaults to earliest prior whole-span table)")
    p.add_argument("--day-folder", help="explicit day folder on the share (defaults to newest Video/<video>/day_* with piece folders)")
    p.add_argument("--tables", action="append", default=[], help="explicit joined-tables folder (repeatable; defaults to discovering all under data/)")
    p.add_argument("--also-capture", action="append", default=[], help="explicit extra capture folder (repeatable; defaults to discovering under data/)")
    p.add_argument("--spots", action="append", default=[], help="explicit joined_clip_spots JSON (repeatable; defaults to discovering under data/<video>/)")
    p.add_argument("--stack-work", help="scratch folder for stack_delivery.py (never on the share)")
    p.add_argument("--start-here", help="optional start_here.json for review_pages.py to-review")
    p.add_argument("--page-url", help="review page URL prefix up to # (defaults to H3_REVIEW_PAGE_URL or '#')")
    p.add_argument("--open-list", help="Open List URL (defaults to H3_OPEN_LIST_URL or '#')")
    p.add_argument("--today", help="owner's day YYYY-MM-DD for review_pages.py to-review")
    p.add_argument("--python", help="python executable for child steps (defaults to sys.executable)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)
    try:
        rebuild(a)
    except Refused as e:
        raise SystemExit(f"REFUSED: {e}") from e
    return 0


if __name__ == "__main__":
    sys.exit(main())

