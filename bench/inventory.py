#!/usr/bin/env python3
"""The inventory of a video: every shot, every subject present in it, what stands and what is owed.

    <python> bench/inventory.py --video <stem> --tables <delivery_tables_dir> --stamp <YYYY-MM-DDTHHMMZ> [--data data] --out <inventory.md> [--json <inventory.json>]

**What it buys.** Reads only the record: `data/<video>/shots.json` (shots, who is present, looks), `data/<video>/laid.jsonl`
(what stands), the rebuild's ledger (`ledger_<video>_<stamp>.json`, for each piece's render day and zoom) and its
frame-coverage file (`frame_coverage_<video>_*_<stamp>.json`, per-frame state). Writes one Markdown table (`--out`) and an
optional structured JSON summary (`--json`). It decides nothing: `done`, `partial`, `nothing laid` and `not seen` are the
record's own states, and whether a face looks right is the playback's question.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

CANNOT_SEE = (
    "whether a face shows on a frame, whether a laid piece looks right on playback, or whether an unseeded tracker "
    "missed a visible subject on a 0 px frame"
)


class Refused(Exception):
    """A missing or mismatched input record refused in the tool's own words with no traceback."""


def spans(frames: list[int]) -> str:
    out: list[list[int]] = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return " ".join(f"{a}-{b}" if a != b else str(a) for a, b in out)


def _read_json(path: Path, label: str):
    if not path.is_file():
        raise Refused(f"REFUSED: {label} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise Refused(f"REFUSED: {label} is not valid JSON ({path}): {e}") from None


def _subject_order(shots: list[dict], plan_j: dict | None) -> list[str]:
    canonical = ["lead", "second", "third"]
    seen: list[str] = []
    if isinstance(plan_j, dict) and isinstance(plan_j.get("subjects"), dict):
        for s in plan_j["subjects"]:
            if s not in seen:
                seen.append(s)
    for r in shots:
        for s in (r.get("who") or {}):
            if s not in seen:
                seen.append(s)
    ordered = [s for s in canonical if s in seen]
    for s in seen:
        if s not in ordered:
            ordered.append(s)
    return ordered or canonical


def _looks_by_subject_frame(shots: list[dict], plan_j: dict | None) -> dict[tuple[str, int], int]:
    plan_parts: dict[str, set[str]] = {}
    if isinstance(plan_j, dict) and isinstance(plan_j.get("subjects"), dict):
        for sub, cfg in plan_j["subjects"].items():
            if isinstance(cfg, dict) and isinstance(cfg.get("parts"), list):
                plan_parts[sub] = {str(p) for p in cfg["parts"]}
    looks: dict[tuple[str, int], int] = {}
    for r in shots:
        for sub, w in (r.get("who") or {}).items():
            if not isinstance(w, dict):
                continue
            sub_parts = plan_parts.get(sub)
            for lk in w.get("seen_looks") or []:
                if not isinstance(lk, dict) or "first" not in lk or "last" not in lk:
                    continue
                part = str(lk.get("part") or "head")
                if sub_parts and part not in ("head", "all") and part not in sub_parts:
                    continue
                for f in range(int(lk["first"]), int(lk["last"]) + 1):
                    if lk.get("is") == "not seen":
                        looks[(sub, f)] = 1
                    else:
                        looks.pop((sub, f), None)
    return looks


def build_inventory(video: str, data_dir: Path, tables_dir: Path, stamp: str) -> dict:
    if not video or not video.strip():
        raise Refused("REFUSED: --video must be non-empty")
    if not stamp or not stamp.strip():
        raise Refused("REFUSED: --stamp must be non-empty")
    if not tables_dir.is_dir():
        raise Refused(f"REFUSED: --tables directory not found: {tables_dir}")

    vdir = data_dir / video
    shots_j = _read_json(vdir / "shots.json", "shots.json")
    shots_raw = shots_j.get("shots") if isinstance(shots_j, dict) and "shots" in shots_j else shots_j
    shots = [
        r for r in (shots_raw if isinstance(shots_raw, list) else (shots_raw.values() if isinstance(shots_raw, dict) else []))
        if isinstance(r, dict) and "first" in r and "last" in r
    ]
    if not shots:
        raise Refused(f"REFUSED: no shots found in {vdir / 'shots.json'}")

    laid_path = vdir / "laid.jsonl"
    if not laid_path.is_file():
        raise Refused(f"REFUSED: laid.jsonl not found: {laid_path}")
    laid_sha256 = hashlib.sha256(laid_path.read_bytes()).hexdigest()
    laid: list[dict] = []
    for line_no, raw in enumerate(laid_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except ValueError as e:
            raise Refused(f"REFUSED: {laid_path} line {line_no} is not valid JSON: {e}") from None
        if isinstance(obj, dict):
            laid.append(obj)

    plan_path = vdir / "plan.json"
    plan_j = _read_json(plan_path, "plan.json") if plan_path.is_file() else None

    table_matches = (
        sorted(tables_dir.glob(f"t_{video}_*_{stamp}.json"))
        or sorted(tables_dir.glob(f"delivery_table_{video}_{stamp}.json"))
    )
    table_path = table_matches[-1] if table_matches else None
    if table_path is not None:
        table_j = _read_json(table_path, "delivery table JSON")
        if isinstance(table_j, dict):
            m_sha = re.search(r"laid\.jsonl[^()]*\(sha256\s+([0-9a-f]+)", str(table_j.get("made") or ""))
            table_laid_sha = (m_sha.group(1) if m_sha else str(table_j.get("laid_sha256") or "")).strip()
            if table_laid_sha and not (
                laid_sha256.startswith(table_laid_sha) or table_laid_sha.startswith(laid_sha256)
            ):
                raise Refused(
                    f"REFUSED: laid.jsonl (sha256 {laid_sha256[:16]}) does not match {table_path.name} "
                    f"(sha256 {table_laid_sha[:16]}): laid.jsonl changed since {stamp}"
                )
            if isinstance(table_j.get("reading"), list):
                stands_by_key: dict[tuple[int, int, str], list[str]] = {}
                for e in table_j["reading"]:
                    if isinstance(e, dict) and "first" in e and "last" in e and "subject" in e:
                        key = (int(e["first"]), int(e["last"]), str(e["subject"]))
                        stands_by_key[key] = sorted(
                            str(st.get("render") or st.get("load") or "")
                            for st in (e.get("stands") or [])
                            if isinstance(st, dict)
                        )
                for (s_first, s_last, s_sub), stands_loads in sorted(stands_by_key.items()):
                    laid_loads = sorted(
                        str(x.get("load") or "")
                        for x in laid
                        if int(x.get("first", -1)) == s_first
                        and int(x.get("last", -1)) == s_last
                        and str(x.get("subject")) == s_sub
                    )
                    if laid_loads != stands_loads:
                        raise Refused(
                            f"REFUSED: laid.jsonl pieces {laid_loads} on {s_first}-{s_last} ({s_sub}) disagree "
                            f"with {table_path.name} reading[].stands {stands_loads}"
                        )
                for x in laid:
                    key = (int(x.get("first", -1)), int(x.get("last", -1)), str(x.get("subject") or ""))
                    if stands_by_key and str(x.get("load") or "") not in stands_by_key.get(key, []):
                        raise Refused(
                            f"REFUSED: laid.jsonl piece {x.get('load')!r} on {key[0]}-{key[1]} ({key[2]}) "
                            f"is absent from {table_path.name} reading[].stands {stands_by_key.get(key, [])}"
                        )

    ledger_path = tables_dir / f"ledger_{video}_{stamp}.json"
    ledger = _read_json(ledger_path, "ledger JSON")
    renders = ledger.get("renders") if isinstance(ledger, dict) else None
    if not isinstance(renders, dict):
        raise Refused(f"REFUSED: ledger JSON missing 'renders' dict: {ledger_path}")
    by_name = {r["name"]: r for r in renders.values() if isinstance(r, dict) and "name" in r}

    cov_matches = sorted(tables_dir.glob(f"frame_coverage_{video}_*_{stamp}.json"))
    if not cov_matches:
        raise Refused(f"REFUSED: no frame_coverage_{video}_*_{stamp}.json found in {tables_dir}")
    cov_path = cov_matches[-1]
    cov = _read_json(cov_path, "frame_coverage JSON")
    if not isinstance(cov, dict) or not isinstance(cov.get("shots"), list):
        raise Refused(f"REFUSED: frame_coverage JSON missing 'shots' list: {cov_path}")

    covmap = {(tuple(s["shot"]), s["subject"]): s for s in cov["shots"] if isinstance(s, dict) and "shot" in s and "subject" in s}
    looks = _looks_by_subject_frame(shots, plan_j)
    subjects = _subject_order(shots, plan_j)

    rows: list[dict] = []
    tot: Counter = Counter()
    for r in sorted(shots, key=lambda x: int(x["first"])):
        first, last = int(r["first"]), int(r["last"])
        for sub in subjects:
            w = (r.get("who") or {}).get(sub) or {}
            if not isinstance(w, dict) or w.get("present") != "yes":
                continue
            pieces = [x for x in laid if int(x.get("first", -1)) == first and x.get("subject") == sub]
            pdesc: list[str] = []
            for x in pieces:
                rec = by_name.get(x.get("load", ""))
                day = str(rec.get("written") or "")[5:10] if rec and rec.get("written") else "?"
                region = str((rec or {}).get("graph", {}).get("render_region", ""))
                zoom = "zoom" if "zoom" in region else ("whole" if rec else "?")
                fr = " ".join(f"{p0}-{p1}" for p0, p1 in (x.get("frames") or [[first, last]]))
                owner = "" if x.get("owner") == "kept" else f", {x.get('owner')}"
                pdesc.append(f"{x.get('load', '?')} ({day}, {zoom}, {fr}{owner})")
            c = covmap.get(((first, last), sub))
            owed: list[int] = []
            left: list[int] = []
            if c:
                states = c.get("state") or [None] * len(c["frames"])
                for f, st, state in zip(c["frames"], c["step"], states):
                    f_int = int(f)
                    s = str(state or "")
                    if s.startswith("left out by a look") or (sub, f_int) in looks:
                        left.append(f_int)
                        continue
                    if int(st) < 4 or "emptied" in s:
                        owed.append(f_int)
            else:
                for f_int in range(first, last + 1):
                    if (sub, f_int) in looks:
                        left.append(f_int)
                    elif not pieces:
                        owed.append(f_int)
            if not owed and pieces:
                state = "done"
            elif not owed and left and not pieces:
                state = "not seen"
            elif not pieces:
                state = "nothing laid"
            else:
                state = "partial"
            tot[state] += 1
            rows.append({
                "first": first,
                "last": last,
                "frames": last - first + 1,
                "subject": sub,
                "state": state,
                "owed": len(owed),
                "owed_spans": spans(owed),
                "by_a_look": len(left),
                "pieces_laid": "; ".join(pdesc) or "-",
            })

    return {
        "video": video,
        "stamp": stamp,
        "measured_on": {
            "shots": str(vdir / "shots.json"),
            "laid": str(laid_path),
            "ledger": str(ledger_path),
            "frame_coverage": str(cov_path),
            **({"delivery_table": str(table_path)} if table_path is not None else {}),
        },
        "cannot_see": CANNOT_SEE,
        "totals": dict(tot),
        "shot_subjects": len(rows),
        "owed_frames": sum(r["owed"] for r in rows),
        "rows": rows,
    }


def format_markdown(inv: dict) -> str:
    video = inv["video"]
    stamp = inv["stamp"]
    tot = inv["totals"]
    rows = inv["rows"]
    lines = [
        f"# Inventory of {video}: every shot, every subject present, what stands and what is owed",
        "",
        f"Made from `shots.json`, `laid.jsonl`, the ledger and the coverage file of the {stamp} rebuild. A reading of that cut;",
        "remade at every rebuild. Cannot see whether a face shows on a frame. State: done = a piece stands on every present frame",
        "not left out by a look; partial = a piece stands and frames are owed; nothing laid = present, no piece; not seen = every",
        "frame left out by a look. Owed = frames where the subject is the original and no look leaves them out.",
        "",
        f"Totals: {tot}; shot-subjects {inv['shot_subjects']}; owed frames {inv['owed_frames']}.",
        "",
        "| shot | frames | subject | state | owed | owed spans | by a look | pieces laid (render day, zoom, frames) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['first']}-{r['last']} | {r['frames']} | {r['subject']} | {r['state']} | "
            f"{r['owed']} | {r['owed_spans']} | {r['by_a_look']} | {r['pieces_laid']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--video", required=True, help="video stem under --data")
    p.add_argument("--data", default="data", help="data root directory (default: data)")
    p.add_argument("--tables", required=True, help="the rebuild's delivery tables folder")
    p.add_argument("--stamp", required=True, help="the rebuild's stamp, e.g. 2026-10-24T2341Z")
    p.add_argument("--out", required=True, help="output Markdown path")
    p.add_argument("--json", dest="json_out", help="optional output JSON path")
    a = p.parse_args(argv)
    try:
        inv = build_inventory(a.video, Path(a.data), Path(a.tables), a.stamp)
    except Refused as e:
        sys.stderr.write(f"{e}\n")
        return 1
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(format_markdown(inv), encoding="utf-8")
    if a.json_out:
        j_path = Path(a.json_out)
        j_path.parent.mkdir(parents=True, exist_ok=True)
        j_path.write_text(json.dumps(inv, indent=2) + "\n", encoding="utf-8")
    print(f"{a.out}: {inv['totals']}, {inv['shot_subjects']} shot-subjects, {inv['owed_frames']} owed frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
