#!/usr/bin/env python3
"""The list of what is laid: one JSON line a piece, the ONLY thing the delivery table reads for what stands.

    <python> bench/laid.py seed --table T.json --share SHARE --out data/<video>/laid.jsonl --by NAME
    <python> bench/laid.py lay --video <video> --first F --last L --subject SUB --load LOAD --file FILE --owner OWNER [--owner-word FILE:LINE] --by WHY [--no-rebuild WHY]
    <python> bench/laid.py replace --video <video> --first F --last L --subject SUB --load LOAD --file FILE --owner OWNER [--owner-word FILE:LINE] --by WHY [--replaces OLD] [--no-rebuild WHY]
    <python> bench/laid.py set-owner --video <video> --load LOAD [--shot F-L] [--subject SUB] --owner OWNER [--owner-word FILE:LINE] --by WHY [--no-rebuild WHY]
    <python> bench/laid.py remove --video <video> [--first F --last L] [--subject SUB] [--load LOAD] --by WHY [--no-rebuild WHY]
    <python> bench/laid.py sync --video <video> [--laid data/<video>/laid.jsonl] [--answers data/<video>/owner_answers.jsonl] [--verdicts data/<video>/verdicts/verdicts.jsonl] [--json OUT.json]
    <python> bench/laid.py verify --laid data/<video>/laid.jsonl --share SHARE
    <python> bench/laid.py keeps --video <video> [--laid data/<video>/laid.jsonl] [--answers data/<video>/owner_answers.jsonl] [--verdicts data/<video>/verdicts/verdicts.jsonl] [--removed data/<video>/laid.removed.jsonl] [--json OUT.json]
    <python> bench/laid.py refs --laid data/<video>/laid.jsonl --ledger LEDGER.json --share SHARE
    (read by) <python> bench/delivery_table.py ... --laid data/<video>/laid.jsonl

**Why** (the owner's order of 2026-10-16, `docs/wiki/decisions.md`): what the cut laid was decided in four places
(the owner's keeps, a session's `prefers.txt`, lines typed into a rebuild script, the newest table by file time),
and what the owner reviewed drifted from what was laid. Here one line says which render is laid on a shot for a
subject, in what order among that shot's lines, the sha256 of its file as laid, and who chose it: `owner` is `kept` where the owner's last choice on that render is a keep (`delivery_table.KEEP`)
given on the render's own row, `kept in a joined clip` where that keep was given on a joined clip that laid it (the
ledger counts it as a keep on each piece, the ruling of 2026-10-14; the owner watched the piece inside the joined
clip, not alone), else `unconfirmed` (mrgan's cold read, 2026-10-16).

**A line** (`schema` `h3_laid/1`): `first`, `last` (the shot, as the shots file has it), `subject`, `load`, `file`
(relative to the share, as the ledger has it), `sha256`, `owner`, `by` (the chooser's words: the owner's keep with
its time and row, or a session's reason), `frames` (the frames the piece is laid on: since 2026-10-21 the delivery
table lays the render on these and no others, says the rest of the shot is left as the source with this line's `by`,
and refuses frames outside the shot or none of the render's; until then they were read nowhere that lays, and
5889e's 5889-5915 was laid on all of 5889-5934), `added`, `added_by`, and when `--no-rebuild WHY` was given,
`no_rebuild`. No two lines in `laid.jsonl` may claim the same `(subject, frame)` (the owner's ruling of 2026-10-30):
when `lay`, `replace`, or `sync` writes a kept line that overlaps an existing kept line of the same subject (on that
shot or across shots), the newer keep by effective keep time replaces every single frame that exists in that clip
(`frames`), regardless of what existed before, carving the overlapping frames out of the older keep's `frames`
(recording `carved_by` on the older row, or moving it to `laid.removed.jsonl` with `carved_whole_by` if no frames
remain, and restoring those frames if the newer keep is later removed). Nothing refuses an owner's keep.

**A removed line leaves its record** (2026-10-21): `remove` takes `--by`, and appends each line it takes out, whole,
to `laid.removed.jsonl` beside the list (`schema` `h3_laid_removed/1`: `removed`, `by`, `at`, `by_session`, the list's
sha256 before and after, and `no_rebuild` when given). Until then the reason was kept nowhere, and `--no-rebuild`'s
was written onto whichever line was last in the list.

**`--owner` is always said** (2026-10-21): `lay` and `replace` take it with no default. Until then it defaulted to
`kept`, and on 2026-10-20 five pieces laid on a go to render read `kept` with no keep behind them. `kept` and `kept in
a joined clip` need `--owner-word FILE:LINE` on every command that writes them, `lay` and `replace` as well as
`set-owner`, and the cited place is kept on the line as `owner_word`. The line is read (counted as `grep -n` counts)
and refused unless it is the owner's (`who` is `owner`, or a line of the verdicts record), its `chosen` is a keep
(`delivery_table.KEEP`, or the bare "keep" the lead's terminal took down once), no owner choice on the same row in
that file with a later `at` (the latest `at`, as `delivery_table.last_choice` takes it, since the verdicts record
holds older lines appended late; the file's order decides between lines of one time, where `max` there takes the
first, and a line with no `at` is the earliest here and the latest there; `at` is compared as text, so a form other
than `YYYY-MM-DDTHH:MM:SSZ` can misorder within a minute or by an offset) is something other than a keep, and its row is this load's
(for `kept`) or a joined clip's, a `joined` part in its name (for `kept in a joined clip`). A line counts as the
owner's by `who` or by the verdicts schema, from whatever file it is cited in. What it cannot see: a later answer in
another file, and whether the joined clip laid this load. `unconfirmed` takes no citation. A reset to `unconfirmed`
of a line whose keep rests on words no record holds as a choice (a spoken look, a hand correction) cannot be undone
by this tool: no line can be cited for it.

**`set-owner`** (2026-10-21) changes one line's `owner` and `by` IN PLACE and nothing else: its place in the file,
`replaces`, `added`, `frames`, `no_rebuild` and the sha256 stand, where a `replace` of a line by itself would write all
of them again. The line is found by `--load`, with `--shot` and `--subject` where that is not one line. Each change is
appended to the line's `history` (`at`, `by_session`, `owner_was`, `by_was`, `owner`, `by`, `owner_word`, and its own
`no_rebuild` reason when given), so what the line said before stays on it; the line's `owner_word` is the current
citation, or absent. Changing to the state the line already has is refused. An earlier hand correction's
`owner_was` or `by_was` at the top of a line is left as it is; `history` is the record from here on.

**Automatic rebuild** (2026-10-18): after every successful write to `data/<video>/laid.jsonl` (`lay`, `replace`,
`remove`, `set-owner`), `laid.py` runs `bench/rebuild_delivery.py --video <video>` without `--clip` and prints its stamp.
That rebuild reuses the newest assembled clip and refuses when the clip was built from another `laid.jsonl`
(`bench/rebuild_delivery.py::verify_reused_clip`), which any write here makes so: a batch takes `--no-rebuild` and one
`rebuild_delivery.py --clip` after it. The pages' "as of" time is the last line's `added`, which a `set-owner` on
another line does not move. If the
rebuild fails, the `laid.jsonl` change still stands, and `laid.py` says so loudly in its own words with the rebuild's
error and exits non-zero. Pass `--no-rebuild WHY` on every write of a batch, the reason required and recorded
(`no_rebuild`, or in the history entry for `set-owner`), and run the one `rebuild_delivery.py --clip` after it.

**Hard failures** (`read`, and so the table): a line that is not JSON or lacks a field, two lines naming one load
on one shot and subject, a file that is missing, a file whose bytes are not the line's sha256. Changing a laid
piece is a new line written with its reason, never an edit that keeps the old sha256.

**`refs`** says, for every line, whether the reference video its render was given is the file at that path now. The
render's build record names it (`files.motion`); where the queue record kept a digest (records queued from
631a9206 on) the bytes are compared (`same bytes as queued` / `CHANGED since queued`); where it did not, only the
file's time is read against the render's (`older than the render`, consistent and not proof; `REWRITTEN after the
render`, the render read something else). Data, not a verdict: a rewritten reference says the record points at the
wrong file, not that the picture is wrong; the original may be under `references/*_SUPERSEDED_*`.

What it cannot see: whether the owner watched these bytes when they kept the render (answers before 2026-10-16 do
not record the clip's sha256), and whether a render read the reference its record names (`bench/laid.py` does not
open the graph).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SCHEMA = "h3_laid/1"
#: A line of `laid.removed.jsonl`: a line taken out of the list, whole, with why and by whom (2026-10-21).
REMOVED_SCHEMA = "h3_laid_removed/1"
FIELDS = ("first", "last", "subject", "load", "file", "sha256", "owner", "by")
OWNER = ("kept", "kept in a joined clip", "unconfirmed")


def removed_path(laid_path: Path) -> Path:
    """Where the lines taken out of a list are kept: `laid.removed.jsonl` beside it."""
    return Path(laid_path).with_name(Path(laid_path).stem + ".removed.jsonl")


def zoomed(f: Path) -> Path:
    """The companion the assembler reads at `--size source` (`assemble_delivery.py`, the zoomed piece)."""
    return f.with_name(f.name.replace("_00001.mp4", "_00001_zoomed.mp4"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(path: str) -> str:
    return sha256(Path(path))


def read(path: str, share: str, check_bytes: bool = True) -> list[dict]:
    """Every line, with its 1-based `line` number; stops on any hard failure named in the docstring."""
    out, seen = [], set()
    for n, text in enumerate(Path(path).read_text().splitlines(), 1):
        if not text.strip():
            continue
        try:
            x = json.loads(text)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}, line {n}: not JSON ({e})")
        missing = [k for k in FIELDS if k not in x]
        if x.get("schema") != SCHEMA or missing:
            raise SystemExit(f"{path}, line {n}: schema {x.get('schema')!r} (wanted {SCHEMA!r}); missing {missing}")
        fr = x.get("frames")
        if fr is not None and not (isinstance(fr, list) and fr and all(
                isinstance(f, list) and len(f) == 2 and all(type(v) is int for v in f) and x["first"] <= f[0] <= f[1] <= x["last"] for f in fr)):
            raise SystemExit(f"{path}, line {n}: frames {fr!r} are not [first, last] spans inside the shot {x['first']}-{x['last']}: "
                             "the delivery table lays the piece on them (since 2026-10-21)")
        if x["owner"] not in OWNER:
            raise SystemExit(f"{path}, line {n}: owner {x['owner']!r} is not one of {OWNER}")
        key = (x["first"], x["last"], x["subject"], x["load"])
        if key in seen:
            raise SystemExit(f"{path}, line {n}: {x['load']} on {x['first']}-{x['last']}, {x['subject']}, is named twice")
        seen.add(key)
        x_fset = _frame_set(_spans_of_laid(x))
        for prev in out:
            if prev["subject"] == x["subject"]:
                ov = _frame_set(_spans_of_laid(prev)) & x_fset
                if ov:
                    raise SystemExit(
                        f"{path}, line {n}: {x['load']} on {x['first']}-{x['last']} ({x['subject']}) overlaps "
                        f"line {prev['line']} ({prev['load']} on {prev['first']}-{prev['last']}) on frame(s) {_fmt_spans(_runs_of_frames(ov))}: "
                        "two rows in laid.jsonl cannot claim the same (subject, frame)"
                    )
        if check_bytes:
            f = Path(share) / x["file"]
            if not f.is_file():
                raise SystemExit(f"{path}, line {n}: {f} is missing")
            now = sha256(f)
            if now != x["sha256"]:
                raise SystemExit(f"{path}, line {n}: {f} is sha256 {now[:16]} now, and the line says {x['sha256'][:16]}: the laid piece changed")
            if x.get("zoomed_sha256") and sha256(zoomed(f)) != x["zoomed_sha256"]:
                raise SystemExit(f"{path}, line {n}: {zoomed(f)} is not the bytes the line says: the laid piece's zoomed companion changed")
        out.append({**x, "line": n})
    return out


def seed(table: str, share: str, by: str) -> list[dict]:
    """The lines that reproduce what a table laid, from its JSON's reading (`stands`), in its order."""
    import delivery_table as dt
    t = json.loads(Path(table).read_text())
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = []
    for e in t["reading"]:
        for st in e["stands"]:
            c = st.get("the_owners_last_choice")
            kept = bool(c and c["said"] in dt.KEEP)
            in_joined = kept and "_joined_" in str(c.get("row") or "")
            if st.get("step") == 0:
                chose = f"a session's --prefer: {st.get('the_leads_prefer')}"
            elif st.get("step") == 1:
                chose = f"the owner's keep of {str(c['at'])[:16]}Z (row {c.get('row')})"
            else:
                chose = "nobody chose it: the old rule's step 2, the newest render in the frozen form" + (
                    f"; the owner's last choice on it: \"{c['said']}\" ({str(c['at'])[:16]}Z)" if c else "; the owner has no choice on it")
            if st.get("step") == 0 and kept:
                chose += f"; and the owner's keep of {str(c['at'])[:16]}Z (row {c.get('row')})"
            lines.append({"schema": SCHEMA, "first": e["first"], "last": e["last"], "subject": e["subject"], "load": st["render"],
                          "file": st["file"], "sha256": sha256(Path(share) / st["file"]), "owner": ("kept in a joined clip" if in_joined else "kept") if kept else "unconfirmed",
                          "by": chose, "frames": st["frames"], "added": now, "added_by": by,
                          "seeded_from": Path(table).name})
    return lines


def refs(lines: list[dict], ledger: dict, repo: Path) -> list[dict]:
    """One row a laid line and reference: what the render's records say it read, against the file now."""
    out = []
    for x in lines:
        r = ledger["renders"].get(x["file"]) or {}                      # the ledger is keyed by the render's file
        rec = next((c for c in reversed(r.get("records") or []) if c.get("built_file")), None)   # the newest build record
        if rec is None:
            out.append({**x, "reference": None, "state": "no build record names a reference"})
            continue
        built = json.loads((repo / rec["built_file"]).read_text())
        queued = json.loads((repo / rec["queued_file"]).read_text()) if rec.get("queued_file") and (repo / rec["queued_file"]).is_file() else {}
        m = (built.get("files") or {}).get("motion") or (built.get("load") or {}).get("motion_file")
        motion = [m] if isinstance(m, str) else [m[0]] if m and isinstance(m[0], str) else [v[0] for v in m or [] if isinstance(v, list)]  # [path, first frame]
        if not motion:
            out.append({**x, "reference": None, "state": "the build record names no reference video" + (f" (its region: {r['form']['replaces'][:60]})" if (r.get("form") or {}).get("replaces") else "")})
        written = datetime.datetime.fromisoformat(str(r.get("written")).replace("Z", "+00:00")).timestamp() if r.get("written") else None
        for ref in motion:
            f = Path(ref)
            kept = (queued.get("digests") or {}).get(ref)
            if not f.is_file():
                state = "MISSING at that path"
            elif kept:
                state = "same bytes as queued" if sha256(f) == kept else "CHANGED since queued"
            elif written is None:
                state = "no digest kept and no render time"
            else:
                state = "no digest kept; older than the render" if f.stat().st_mtime <= written else "no digest kept; REWRITTEN after the render"
            found = None
            if kept and state == "CHANGED since queued":
                # the bytes it read, where a remake set the original aside (references/<name>_SUPERSEDED_*/ the same file name)
                found = next((str(g) for g in sorted(f.parent.parent.glob(f"{f.parent.name}_SUPERSEDED_*/{f.name}")) if sha256(g) == kept), None)
                state += f"; the bytes it read are at {found}" if found else "; the bytes it read are not found beside it"
            out.append({**x, "reference": ref, "state": state, "read_at": found})
    return out


def _resolve_laid_and_video(a: argparse.Namespace) -> tuple[Path, str]:
    video = (getattr(a, "video", None) or "").strip()
    laid_arg = (getattr(a, "laid", None) or "").strip()
    data_arg = (getattr(a, "data", None) or "").strip()
    if not video and not laid_arg:
        raise SystemExit(f"{a.cmd}: either --video <video> or --laid data/<video>/laid.jsonl is required")
    if laid_arg:
        laid_path = Path(laid_arg)
        if not video:
            video = laid_path.resolve().parent.name
    elif data_arg:
        dp = Path(data_arg)
        vdir = dp if dp.name == video and not (dp / video).is_dir() else (dp / video)
        laid_path = vdir / "laid.jsonl"
    else:
        laid_path = HERE.parent / "data" / video / "laid.jsonl"
    return laid_path, video


def _resolve_share(a: argparse.Namespace, laid_path: Path) -> Path:
    sh = (getattr(a, "share", None) or os.environ.get("H3_OUTPUT_DIR") or "").strip()
    if sh:
        return Path(sh)
    for cand in filter(None, [getattr(a, "inputs", None), str(laid_path.resolve().parent / "inputs.json")]):
        p = Path(cand)
        if p.is_file():
            try:
                cfg = json.loads(p.read_text())
                if cfg.get("share") or cfg.get("output_dir"):
                    return Path(str(cfg.get("share") or cfg.get("output_dir")))
            except (OSError, ValueError):
                pass
    raise SystemExit(f"{a.cmd}: --share is required (or set H3_OUTPUT_DIR)")


def _parse_shot(a: argparse.Namespace, required: bool = True) -> tuple[int | None, int | None]:
    first, last = getattr(a, "first", None), getattr(a, "last", None)
    shot = (getattr(a, "shot", None) or "").strip()
    if shot:
        m = re.match(r"^(\d+)-(\d+)$", shot)
        if not m:
            raise SystemExit(f"{a.cmd}: --shot {shot!r} must be FIRST-LAST")
        sf, sl = int(m.group(1)), int(m.group(2))
        if (first is not None and int(first) != sf) or (last is not None and int(last) != sl):
            raise SystemExit(f"{a.cmd}: --shot {shot} disagrees with --first {first} / --last {last}")
        first, last = sf, sl
    if required and (first is None or last is None):
        raise SystemExit(f"{a.cmd}: --first and --last (or --shot FIRST-LAST) are required")
    if (first is None) != (last is None):
        raise SystemExit(f"{a.cmd}: give both --first and --last, or neither")
    if first is not None and last is not None and int(first) > int(last):
        raise SystemExit(f"{a.cmd}: --first ({first}) > --last ({last})")
    return (int(first), int(last)) if first is not None and last is not None else (None, None)


def _parse_frames(raw: str | None, first: int, last: int) -> list[list[int]]:
    if not raw:
        return [[first, last]]
    s = raw.strip()
    if s.startswith("["):
        val = json.loads(s)
        return [[int(x[0]), int(x[1])] for x in val]
    out = []
    for part in re.split(r"[,;]+", s):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)-(\d+)$", part)
        if not m:
            raise SystemExit(f"--frames {raw!r}: expected JSON [[first, last], ...] or FIRST-LAST")
        out.append([int(m.group(1)), int(m.group(2))])
    return out or [[first, last]]


def _rel_to_share(file_arg: str, share: Path) -> str:
    p = Path(file_arg)
    if p.is_absolute():
        try:
            return str(p.resolve().relative_to(share.resolve()))
        except ValueError:
            pass
    return file_arg


def _build_entry(a: argparse.Namespace, share: Path, first: int, last: int, why_no_rebuild: str | None, replaces: list[str] | None = None) -> dict:
    if not (a.by or "").strip():
        raise SystemExit(f"{a.cmd}: --by is required (the chooser's reason)")
    if a.owner == "unconfirmed" and getattr(a, "owner_word", None):
        raise SystemExit(f"{a.cmd}: --owner-word cites a keep, and `unconfirmed` is no keep")
    if a.owner != "unconfirmed" and not getattr(a, "owner_word", None):
        raise SystemExit(f"{a.cmd}: `{a.owner}` takes --owner-word FILE:LINE, the owner's own line saying the keep")
    cite = owner_word(a.owner_word, a.load, a.owner, a.cmd) if a.owner != "unconfirmed" else None
    rel_file = _rel_to_share(a.file, share)
    f = share / rel_file
    if not f.is_file():
        raise SystemExit(f"{a.cmd}: {f} is missing")
    now = a.added or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    added_by = (a.added_by or os.environ.get("H3_SESSION") or a.by).strip()
    frames = _parse_frames(a.frames, first, last)
    if not (isinstance(frames, list) and frames and all(
            isinstance(s, list) and len(s) == 2 and all(type(v) is int for v in s) and first <= s[0] <= s[1] <= last
            for s in frames)):
        raise SystemExit(f"{a.cmd}: frames {frames!r} are not [first, last] spans inside the shot {first}-{last}")
    entry: dict = {
        "schema": SCHEMA,
        "first": first,
        "last": last,
        "subject": a.subject,
        "load": a.load,
        "file": rel_file,
        "sha256": sha256(f),
        "owner": a.owner,
        "by": a.by.strip(),
        "frames": frames,
        "added": now,
        "added_by": added_by,
    }
    if cite:
        entry["owner_word"] = cite
    if zoomed(f).is_file():
        entry["zoomed_sha256"] = sha256(zoomed(f))
    if replaces:
        entry["replaces"] = replaces[0] if len(replaces) == 1 else replaces
    if why_no_rebuild:
        entry["no_rebuild"] = why_no_rebuild
    return entry


def _run_autorebuild(a: argparse.Namespace, laid_path: Path, video: str, share: Path) -> int:
    cmd = [sys.executable, str(HERE / "rebuild_delivery.py"), "--video", video]
    default_vdir = (HERE.parent / "data" / video).resolve()
    laid_dir = laid_path.resolve().parent
    data_root: Path | None = None
    if getattr(a, "data", None):
        dp = Path(a.data).resolve()
        data_root = dp.parent if dp.name == video and not (dp / video).is_dir() else dp
        cmd += ["--data", str(data_root)]
    elif laid_dir != default_vdir and laid_dir.name == video and laid_dir.parent.is_dir():
        data_root = laid_dir.parent
        cmd += ["--data", str(data_root)]
    if getattr(a, "inputs", None):
        cmd += ["--inputs", str(a.inputs)]
    elif data_root is not None and (data_root.parent / "inputs").is_dir():
        cmd += ["--inputs", str(data_root.parent / "inputs")]
    if getattr(a, "laid", None) or laid_dir != default_vdir:
        cmd += ["--laid", str(laid_path)]
    if getattr(a, "share", None):
        cmd += ["--share", str(share)]
    for flag, attr in (("--out-dir", "out_dir"), ("--day", "day"), ("--stamp", "stamp"), ("--source", "source")):
        val = getattr(a, attr, None)
        if val:
            cmd += [flag, str(val)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        err = (res.stderr or res.stdout or "").strip()
        msg = (
            f"LAID STANDS, REBUILD FAILED: {laid_path} was written and the laid.jsonl change still stands, "
            f"but bench/rebuild_delivery.py --video {video} failed (exit {res.returncode}): {err}"
        )
        raise SystemExit(msg)
    out_text = (res.stdout or "").strip()
    m = re.search(r"^rebuild\s+\S+\s+(\S+)", out_text, re.M)
    stamp = m.group(1) if m else (getattr(a, "stamp", None) or "")
    if out_text:
        print(out_text)
    if stamp:
        print(f"rebuild stamp: {stamp}")
    return 0


def _runs_of_frames(nums: list[int] | set[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for n in sorted({int(x) for x in nums}):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return runs


def _not_seen_frames_of_shot(
    laid_path: Path, video: str, data_arg: str | None, first: int, last: int, subject: str
) -> set[int]:
    """Return the set of frames in `[first, last]` covered by `is: 'not seen'` in `shots.json` for `subject`."""
    candidates: list[Path] = [laid_path.resolve().parent / "shots.json"]
    if data_arg:
        dp = Path(data_arg).resolve()
        candidates.append(dp / "shots.json")
        if video:
            candidates.append(dp / video / "shots.json")
    if video:
        candidates.append(HERE.parent / "data" / video / "shots.json")
    shots_file = next((p for p in candidates if p.is_file()), None)
    if shots_file is None:
        return set()
    try:
        shots_data = json.loads(shots_file.read_text())
    except Exception:
        return set()
    subj_parts: set[str] = set()
    plan_file = shots_file.parent / "plan.json"
    if plan_file.is_file():
        try:
            plan_data = json.loads(plan_file.read_text())
            parts_cfg = ((plan_data.get("subjects") or {}).get(subject) or {}).get("parts") or []
            if isinstance(parts_cfg, list):
                subj_parts = {str(p) for p in parts_cfg}
        except Exception:
            pass
    shot_frames = set(range(first, last + 1))
    for s in shots_data.get("shots") or []:
        if int(s.get("first", -1)) == first and int(s.get("last", -1)) == last:
            raw_looks = (((s.get("who") or {}).get(subject) or {}).get("seen_looks")) or []
            seen_looks = sorted(
                [lk for lk in raw_looks if isinstance(lk, dict)],
                key=lambda x: str(x.get("at") or ""),
            )
            if not seen_looks:
                return set()
            pts = sorted(subj_parts) if subj_parts else ["head"]
            unseen_per_part: list[set[int]] = []
            for pt in pts:
                unseen_pt: set[int] = set()
                for lk in seen_looks:
                    part = str(lk.get("part") or "head")
                    if subj_parts and part not in (pt, "head", "whole", "all"):
                        continue
                    if "first" in lk and "last" in lk:
                        f0, f1 = int(lk["first"]), int(lk["last"])
                    elif isinstance(lk.get("frames"), list) and len(lk["frames"]) == 2:
                        f0, f1 = int(lk["frames"][0]), int(lk["frames"][1])
                    else:
                        continue
                    span_set = set(range(max(f0, first), min(f1, last) + 1)) & shot_frames
                    if lk.get("is") == "not seen":
                        unseen_pt |= span_set
                    elif lk.get("is") == "seen":
                        unseen_pt -= span_set
                unseen_per_part.append(unseen_pt)
            return set.intersection(*unseen_per_part) if unseen_per_part else set()
    return set()


def _search_data_dirs(laid_path: Path, video: str, data_arg: str | None) -> list[Path]:
    raw: list[Path] = []
    laid_dir = laid_path.resolve().parent
    raw.extend([laid_dir, laid_dir.parent])
    if data_arg:
        dp = Path(data_arg).resolve()
        raw.extend([dp, dp.parent])
        if video:
            raw.append(dp / video)
    if video:
        raw.extend([HERE.parent / "data" / video, HERE.parent / "data"])
    out: list[Path] = []
    seen: set[Path] = set()
    for d in raw:
        if d.is_dir() and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _find_load_job_source(laid_path: Path, video: str, data_arg: str | None, load_name: str) -> str | None:
    """Locate the job JSON for `load_name` under `--data` or beside `laid_path` and return `job.source` if present."""
    candidates: list[Path] = []
    for d in _search_data_dirs(laid_path, video, data_arg):
        for jdir in sorted(d.glob(f"*_{load_name}_job")):
            for jname in ("job.render.json", "job.plan.json", "job.json", f"{load_name}.render.built.json"):
                if (jdir / jname).is_file():
                    candidates.append(jdir / jname)
        for p in (d / "jobs" / f"{load_name}.json", d / f"{load_name}.json"):
            if p.is_file():
                candidates.append(p)
    for jf in candidates:
        try:
            jd = json.loads(jf.read_text())
            src = (
                ((jd.get("job") or {}).get("source") if isinstance(jd.get("job"), dict) else None)
                or jd.get("source")
                or ((jd.get("files") or {}).get("source") if isinstance(jd.get("files"), dict) else None)
            )
            if src:
                return str(src)
        except Exception:
            continue
    return None


def _find_cut_sidecar(
    job_src: str, share: Path | None, video: str, laid_path: Path | None = None
) -> tuple[bool, Path | None, list[dict] | None]:
    """Return `(is_cut_name, sidecar_path_or_None, pieces_or_None)` for `job_src`."""
    src_p = Path(job_src)
    is_cut = any(tag in src_p.name for tag in ("_joined_", "_cut_", "_full_"))
    sidecars: list[Path] = [Path(f"{src_p}.json")]
    if src_p.suffix:
        sidecars.append(src_p.with_suffix(".json"))
    if share is not None:
        sidecars.extend([
            share / f"{src_p}.json",
            share / src_p.with_suffix(src_p.suffix + ".json"),
            share / "Video" / video / f"{src_p.name}.json",
        ])
        m_day = re.search(r"joined_(\d{4}-\d{2}-\d{2})T", src_p.name)
        if m_day:
            sidecars.append(share / "Video" / video / f"day_{m_day.group(1)}" / f"{src_p.name}.json")
    inp_dir = (os.environ.get("H3_INPUT_DIR") or "").strip()
    if inp_dir:
        sidecars.append(Path(inp_dir) / f"{src_p.name}.json")
    if laid_path is not None:
        ld = laid_path.resolve().parent
        sidecars.extend([ld / f"{src_p.name}.json", ld.parent / f"{src_p.name}.json"])
    seen: set[Path] = set()
    for sc in sidecars:
        if sc in seen:
            continue
        seen.add(sc)
        if sc.is_file():
            try:
                doc = json.loads(sc.read_text())
            except Exception:
                continue
            pieces = doc.get("pieces") if isinstance(doc, dict) else None
            if isinstance(pieces, list):
                return True, sc, pieces
    return is_cut, None, None


def _piece_load_name(p: dict) -> str:
    ld = p.get("load")
    if isinstance(ld, dict):
        return str(ld.get("load") or p.get("name") or "")
    return str(ld or p.get("name") or "")


def _source_holds_other_subject(
    job_src: str,
    share: Path,
    video: str,
    other_subject: str,
    overlap_frames: set[int],
    laid_path: Path | None = None,
    other_load: str | None = None,
) -> bool | str:
    """Return True if `job_src` is a cut whose `.json` sidecar holds `other_subject` (and `other_load` when given)
    covering `overlap_frames`, `"cannot_tell"` if `job_src` is a cut with no readable `.json` sidecar `pieces`,
    or False otherwise."""
    is_cut, _sc, pieces = _find_cut_sidecar(job_src, share, video, laid_path)
    if pieces is not None:
        covered: set[int] = set()
        for p in pieces:
            if str(p.get("subject")) != other_subject:
                continue
            if other_load is not None and _piece_load_name(p) != other_load:
                continue
            pf, pl = int(p.get("first", -1)), int(p.get("last", -2))
            if pf <= pl:
                covered.update(range(pf, pl + 1))
        return overlap_frames.issubset(covered)
    if is_cut:
        return "cannot_tell"
    return False


def _norm_iso(ts: str) -> str:
    s = str(ts or "").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z$", s):
        return s[:16] + ":00Z"
    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2}$", s):
        try:
            dt = datetime.datetime.fromisoformat(s).astimezone(datetime.timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    return s


def _load_owner_choices(
    laid_path: Path,
    answers_arg: str | None = None,
    verdicts_arg: str | None = None,
    extra_cites: list[str] | None = None,
) -> tuple[dict[str, dict], dict[str, list[list[int]]]]:
    """Build `(choices, frames_by_row)` from `verdicts.jsonl`, `owner_answers.jsonl`, and any cited `--owner-word` files.
    When an answer appears in both `verdicts.jsonl` and `owner_answers.jsonl` with the same `at`, `owner_answers.jsonl` wins."""
    import delivery_table

    vdir = laid_path.resolve().parent
    answers_path = Path(answers_arg) if answers_arg else (vdir / "owner_answers.jsonl")
    if verdicts_arg:
        verdicts_path = Path(verdicts_arg)
    else:
        cand = vdir / "verdicts" / "verdicts.jsonl"
        verdicts_path = cand if cand.is_file() else (vdir / "verdicts.jsonl")

    files_to_read: list[tuple[int, Path, bool]] = [(0, verdicts_path, False), (1, answers_path, True)]
    seen_files: set[Path] = {p.resolve() for _, p, _ in files_to_read if p.is_file()}
    for cite in extra_cites or []:
        cpath_str, sep, _ = str(cite or "").rpartition(":")
        if sep and cpath_str:
            cp = Path(cpath_str)
            if not cp.is_file() and cpath_str.startswith(f"data/{vdir.name}/"):
                cand_cp = vdir / cpath_str[len(f"data/{vdir.name}/"):]
                if cand_cp.is_file():
                    cp = cand_cp
            if not cp.is_file() and (HERE.parent / cpath_str).is_file():
                cp = HERE.parent / cpath_str
            if cp.is_file() and cp.resolve() not in seen_files:
                seen_files.add(cp.resolve())
                files_to_read.append((len(files_to_read), cp, False))

    choices: dict[str, dict] = {}
    frames_by_row: dict[str, list[list[int]]] = {}
    not_the_one_lower = {k.lower() for k in delivery_table.NOT_THE_ONE} | {"not sure", "not this one", "taken back"}
    for f_idx, p, is_ans_file in files_to_read:
        if not p.is_file():
            continue
        rel_cite_prefix = str(p)
        if vdir.parent.name == "data" or (HERE.parent / "data" / vdir.name).is_dir():
            try:
                cand_rel = Path("data") / vdir.name / p.resolve().relative_to(vdir)
                if cand_rel.is_file() or vdir.parent.name == "data" or (HERE.parent / cand_rel).is_file():
                    rel_cite_prefix = str(cand_rel)
            except Exception:
                pass
        for l_idx, text in enumerate(p.read_text().splitlines(), 1):
            if not text.strip():
                continue
            try:
                d = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(d, dict):
                continue
            is_owner = (d.get("who") == "owner") if is_ans_file else (d.get("schema") == "h3_verdicts/1" or d.get("who") == "owner")
            if not is_owner:
                continue
            row = str(d.get("row") or "").strip()
            if not row:
                continue
            fr = d.get("frames")
            if isinstance(fr, list) and len(fr) == 2 and all(type(v) is int for v in fr):
                frames_by_row[row] = [[int(fr[0]), int(fr[1])]]
            elif isinstance(fr, list) and fr and all(isinstance(s, list) and len(s) == 2 for s in fr):
                frames_by_row[row] = [[int(s[0]), int(s[1])] for s in fr]
            elif d.get("first") is not None and fr is not None and str(d.get("first")).isdigit() and str(fr).isdigit():
                f0 = int(d["first"])
                frames_by_row[row] = [[f0, f0 + int(fr) - 1]]
            chosen = str(d.get("chosen") or "").strip()
            if not chosen or d.get("what") == "parts":
                continue
            if d.get("what") == "words" and not (
                _is_keep(chosen) or chosen.lower() in not_the_one_lower or "wrong" in chosen.lower() or "skip" in chosen.lower()
            ):
                continue
            key = (_norm_iso(str(d.get("at") or "")), f_idx, l_idx)
            prev = choices.get(row)
            if prev is None or key > prev["key"]:
                choices[row] = {
                    "key": key,
                    "chosen": chosen,
                    "at": str(d.get("at") or ""),
                    "cite": f"{p.name}:{l_idx}",
                    "cite_path": f"{rel_cite_prefix}:{l_idx}",
                    "what": d.get("what"),
                }

    for row in list(choices.keys()):
        if choices[row]["chosen"].lower() == "taken back":
            del choices[row]
    return choices, frames_by_row


def _is_active_keep(row: dict, choices: dict[str, dict]) -> bool:
    """Return True iff `row` has `owner` in `('kept', 'kept in a joined clip')` and its last owner choice is not a non-keep."""
    if row.get("owner") not in ("kept", "kept in a joined clip"):
        return False
    ld = str(row.get("load") or "")
    ch = choices.get(ld)
    if ch is not None and not _is_keep(ch["chosen"]):
        return False
    return True


def _left_by_load_frames(laid_path: Path, video: str, data_arg: str | None, row: dict) -> tuple[set[int], str]:
    """Query `render_ledger.left_by_the_load(capture, subject, span, run)` directly from the load's capture folder on disk.
    Never falls back to `render_ledger.json`. Returns `(source_frames_set, 'known' | 'not_known')`."""
    import render_ledger

    load_name = str(row.get("load") or "")
    subject = str(row.get("subject") or "")
    if not load_name or not subject:
        return set(), "not_known"

    search_dirs = _search_data_dirs(laid_path, video, data_arg)
    cap_path: Path | None = None
    span: list[int] | None = None

    for d in search_dirs:
        for jdir in sorted(d.glob(f"*_{load_name}_job")):
            for jname in (f"{load_name}.render.built.json", "job.render.json", "job.plan.json", "job.json"):
                jf = jdir / jname
                if not jf.is_file():
                    continue
                try:
                    jd = json.loads(jf.read_text())
                except Exception:
                    continue
                ld_obj = jd.get("load") if isinstance(jd.get("load"), dict) else {}
                if ld_obj.get("first") is not None and ld_obj.get("frames"):
                    f0 = int(ld_obj["first"])
                    span = [f0, f0 + int(ld_obj["frames"]) - 1]
                cap_raw = jd.get("capture") or ((jd.get("job") or {}).get("capture") if isinstance(jd.get("job"), dict) else None)
                if cap_raw and cap_path is None:
                    cp = Path(str(cap_raw))
                    for base in ([cp] if cp.is_absolute() else [d, d.parent, HERE.parent]):
                        cand = base if cp.is_absolute() else (base / cp)
                        if (cand / "manifest.json").is_file():
                            cap_path = cand.resolve()
                            break
        if cap_path is not None:
            break

    if cap_path is None:
        for d in search_dirs:
            for cand in [*sorted(d.glob(f"*_{load_name}")), d / load_name]:
                if (cand / "manifest.json").is_file():
                    cap_path = cand.resolve()
                    break
            if cap_path is not None:
                break

    if cap_path is None or not (cap_path / "manifest.json").is_file():
        return set(), "not_known"

    if span is None:
        try:
            mdoc = json.loads((cap_path / "manifest.json").read_text())
            if mdoc.get("first_frame") is not None and mdoc.get("frames"):
                f0 = int(mdoc["first_frame"])
                span = [f0, f0 + int(mdoc["frames"]) - 1]
        except Exception:
            pass
    if span is None:
        span = [int(row["first"]), int(row["last"])]

    res = render_ledger.left_by_the_load(cap_path, subject, span, load_name)
    if res is None:
        return set(), "not_known"
    return _frame_set(res.get("source_frames") or []), "known"


def _replaces_list(obj: dict) -> list[str]:
    rep = obj.get("replaces")
    if isinstance(rep, list):
        return [str(x).strip() for x in rep if str(x).strip()]
    if isinstance(rep, str) and rep.strip():
        return [rep.strip()]
    return []


def _merge_replaces(row: dict, replaced_loads: list[str]) -> None:
    if not replaced_loads:
        return
    ordered = _replaces_list(row)
    for ld in replaced_loads:
        s = str(ld).strip()
        if s and s not in ordered and s != str(row.get("load") or ""):
            ordered.append(s)
    if ordered:
        row["replaces"] = ordered[0] if len(ordered) == 1 else ordered


def _merge_carved_by(row: dict, by_load: str, spans: list[list[int]], at: str, reason: str) -> None:
    cur = list(row.get("carved_by") or [])
    for entry in cur:
        if entry.get("load") == by_load:
            merged = _frame_set(entry.get("frames") or []) | _frame_set(spans)
            entry["frames"] = _runs_of_frames(merged)
            entry["at"] = at
            entry["reason"] = reason
            row["carved_by"] = cur
            return
    cur.append({"load": by_load, "frames": spans, "at": at, "reason": reason})
    row["carved_by"] = cur


def _merge_original_restored(row: dict, older_load: str, spans: list[list[int]], why: str) -> None:
    cur = list(row.get("original_restored_over_replacement") or [])
    for entry in cur:
        if entry.get("older_load") == older_load:
            merged = _frame_set(entry.get("frames") or []) | _frame_set(spans)
            entry["frames"] = _runs_of_frames(merged)
            entry["why"] = why
            row["original_restored_over_replacement"] = cur
            return
    cur.append({"older_load": older_load, "frames": spans, "why": why})
    row["original_restored_over_replacement"] = cur


def _trim_original_restored(row: dict, carved_fset: set[int]) -> None:
    if not row.get("original_restored_over_replacement"):
        return
    kept_entries: list[dict] = []
    for entry in row["original_restored_over_replacement"]:
        rem = _frame_set(entry.get("frames") or []) - carved_fset
        if rem:
            kept_entries.append({**entry, "frames": _runs_of_frames(rem)})
    if kept_entries:
        row["original_restored_over_replacement"] = kept_entries
    else:
        row.pop("original_restored_over_replacement", None)


def _restore_carved_by_removed(
    new_lines: list[dict], removed_rows: list[dict], laid_path: Path
) -> list[tuple[dict, str]]:
    """Restore frames carved by `removed_rows` on standing rows in `new_lines` and on `carved_whole_by` rows in `laid.removed.jsonl`.
    Returns a list of `(restored_row, removed_load_name)` for rows restored from `laid.removed.jsonl`."""
    restored_from_whole: list[tuple[dict, str]] = []
    rem_file = removed_path(laid_path)
    rem_records: list[dict] = []
    if rem_file.is_file():
        for text in rem_file.read_text().splitlines():
            if text.strip():
                try:
                    rem_records.append(json.loads(text))
                except Exception:
                    continue

    for rem_r in removed_rows:
        r_load = str(rem_r["load"])
        r_sub = str(rem_r["subject"])

        # 1. Restore carved frames on rows still standing in new_lines
        for row in new_lines:
            if row["subject"] != r_sub or not row.get("carved_by"):
                continue
            matching = [cb for cb in row["carved_by"] if cb.get("load") == r_load]
            if not matching:
                continue
            carved_fset = _frame_set([s for cb in matching for s in (cb.get("frames") or [])])
            other_claimed = set().union(
                *[_frame_set(_spans_of_laid(o)) for o in new_lines if o is not row and o["subject"] == r_sub]
            ) if any(o is not row and o["subject"] == r_sub for o in new_lines) else set()
            restore_fset = carved_fset - other_claimed
            if restore_fset:
                merged_fset = _frame_set(_spans_of_laid(row)) | restore_fset
                row["frames"] = _runs_of_frames(merged_fset)
            remaining_cb = [cb for cb in row["carved_by"] if cb.get("load") != r_load]
            if remaining_cb:
                row["carved_by"] = remaining_cb
            else:
                row.pop("carved_by", None)

        # 2. Restore rows that were carved whole into laid.removed.jsonl (and not later explicitly removed or already restored)
        latest_state: dict[tuple[str, str, int, int], dict] = {}
        for rec in rem_records:
            if isinstance(rec.get("removed"), dict):
                rm = rec["removed"]
                k = (str(rm.get("subject") or ""), str(rm.get("load") or ""), int(rm.get("first", -1)), int(rm.get("last", -1)))
                latest_state[k] = rec
            elif isinstance(rec.get("restored"), dict):
                rs = rec["restored"]
                k = (str(rs.get("subject") or ""), str(rs.get("load") or ""), int(rs.get("first", -1)), int(rs.get("last", -1)))
                latest_state.pop(k, None)

        for (k_sub, k_load, k_first, k_last), rec in list(latest_state.items()):
            if k_sub != r_sub or not rec.get("carved_whole_by"):
                continue
            if any(x["subject"] == k_sub and x["load"] == k_load and int(x["first"]) == k_first and int(x["last"]) == k_last for x in new_lines):
                continue
            old_row = json.loads(json.dumps(rec["removed"]))
            cby = list(old_row.get("carved_by") or [])
            matching = [cb for cb in cby if cb.get("load") == r_load]
            if not matching and rec.get("carved_whole_by") != r_load:
                continue
            carved_fset = (
                _frame_set([s for cb in matching for s in (cb.get("frames") or [])])
                if matching
                else _frame_set(_spans_of_laid(old_row))
            )
            other_claimed = set().union(
                *[_frame_set(_spans_of_laid(o)) for o in new_lines if o["subject"] == r_sub]
            ) if any(o["subject"] == r_sub for o in new_lines) else set()
            restore_fset = carved_fset - other_claimed
            if not restore_fset:
                continue
            old_row["frames"] = _runs_of_frames(restore_fset)
            remaining_cb = [cb for cb in cby if cb.get("load") != r_load]
            if remaining_cb:
                old_row["carved_by"] = remaining_cb
            else:
                old_row.pop("carved_by", None)
            new_lines.append(old_row)
            restored_from_whole.append((json.loads(json.dumps(old_row)), r_load))
    return restored_from_whole


def _carve_same_subject_overlaps(
    candidate_lines: list[dict],
    entry: dict,
    choices: dict[str, dict],
    laid_path: Path,
    video: str,
    data_arg: str | None,
    now: str,
    cmd: str,
) -> tuple[list[dict], list[tuple[dict, str, list[list[int]]]]]:
    """Resolve frame-level overlaps between `entry` and any existing row in `candidate_lines` of the same `subject`.
    When a clip is kept, it replaces every single frame in its clip for that subject (`ov = r_fset & e_fset`),
    decided by effective keep time (newer keep wins). If the newer keep leaves frames `B <= ov` as the original
    where the older keep replaced `subject`, `B` is still won by the newer keep and recorded on the winner under
    `original_restored_over_replacement`.
    Returns `(new_lines, carved_whole_records)` where each item in `carved_whole_records` is `(removed_row, winner_load, carved_spans)`."""
    carved_whole: list[tuple[dict, str, list[list[int]]]] = []
    entry_kept = _is_active_keep(entry, choices)
    surviving_existing: list[dict] = []

    for r in candidate_lines:
        if r["subject"] != entry["subject"]:
            surviving_existing.append(r)
            continue
        r_spans_before = _spans_of_laid(r)
        r_fset = _frame_set(r_spans_before)
        e_fset = _frame_set(_spans_of_laid(entry))
        ov = r_fset & e_fset
        if not ov:
            surviving_existing.append(r)
            continue
        r_kept = _is_active_keep(r, choices)
        if r_kept and not entry_kept:
            raise SystemExit(
                f"{cmd}: {entry['load']} ({entry['owner']}) cannot carve or replace active keep {r['load']} "
                f"on {entry['subject']} frame(s) {_fmt_spans(_runs_of_frames(ov))}"
            )
        if not r_kept and not entry_kept:
            raise SystemExit(
                f"{cmd}: {entry['load']} overlaps existing row {r['load']} on {entry['subject']} frame(s) "
                f"{_fmt_spans(_runs_of_frames(ov))}: use `replace` to replace an unconfirmed row"
            )
        if entry_kept and not r_kept:
            winner, loser = entry, r
        else:
            r_at = _laid_effective_at(r, choices)
            e_at = _laid_effective_at(entry, choices)
            entry_is_newer = e_at >= r_at
            winner = entry if entry_is_newer else r
            loser = r if entry_is_newer else entry
            w_left, w_status = _left_by_load_frames(laid_path, video, data_arg, winner)
            if w_status == "not_known":
                winner["leaves_as_the_original_status"] = "not_known"
            if ov & w_left:
                l_left, l_status = _left_by_load_frames(laid_path, video, data_arg, loser)
                if l_status == "not_known":
                    loser["leaves_as_the_original_status"] = "not_known"
                restored_orig = (ov & w_left) - l_left
                if restored_orig:
                    _merge_original_restored(
                        winner,
                        str(loser["load"]),
                        _runs_of_frames(restored_orig),
                        "newer keep leaves frame(s) as original where older keep replaced subject",
                    )
                for prev_or in loser.get("original_restored_over_replacement") or []:
                    prev_ov = _frame_set(prev_or.get("frames") or []) & (ov & w_left)
                    if prev_ov:
                        _merge_original_restored(
                            winner,
                            str(prev_or.get("older_load") or loser["load"]),
                            _runs_of_frames(prev_ov),
                            str(prev_or.get("why") or "newer keep leaves frame(s) as original where older keep replaced subject"),
                        )

        l_before = _spans_of_laid(loser)
        carved_spans = _runs_of_frames(ov)
        w_eff = _laid_effective_at(winner, choices) or now
        _merge_carved_by(
            loser,
            str(winner["load"]),
            carved_spans,
            w_eff,
            f"carved by newer keep {winner['load']} ({w_eff})",
        )
        _trim_original_restored(loser, ov)
        l_rem = _frame_set(l_before) - ov
        if l_rem:
            loser["frames"] = _runs_of_frames(l_rem)
            surviving_existing.append(r)
        else:
            if loser is entry:
                raise SystemExit(
                    f"{cmd}: {entry['load']} (keep at {_laid_effective_at(entry, choices)}) is older than standing keep "
                    f"{winner['load']} ({_laid_effective_at(winner, choices)}) on all of its frames {_fmt_spans(carved_spans)}"
                )
            if (int(loser["first"]), int(loser["last"])) == (int(winner["first"]), int(winner["last"])):
                _merge_replaces(winner, [str(loser["load"]), *_replaces_list(loser)])
            loser["frames"] = l_before
            carved_whole.append((loser, str(winner["load"]), carved_spans))

    return [*surviving_existing, entry], carved_whole


def mutate_laid(a: argparse.Namespace) -> int:
    why_no_rebuild: str | None = None
    if a.no_rebuild is not None:
        why_no_rebuild = a.no_rebuild.strip()
        if not why_no_rebuild or why_no_rebuild.startswith("-"):
            raise SystemExit("--no-rebuild requires a non-empty reason (WHY), recorded in the line, or for remove in laid.removed.jsonl")
    why_partial_lay: str | None = None
    if getattr(a, "allow_partial_lay", None) is not None:
        why_partial_lay = a.allow_partial_lay.strip()
        if not why_partial_lay or why_partial_lay.startswith("-"):
            raise SystemExit("--allow-partial-lay requires a non-empty reason (WHY), recorded in the line as partial_lay_why")
    why_parallel_raw: str | None = None
    if getattr(a, "allow_parallel_raw", None) is not None:
        why_parallel_raw = a.allow_parallel_raw.strip()
        if not why_parallel_raw or why_parallel_raw.startswith("-"):
            raise SystemExit("--allow-parallel-raw requires a non-empty reason (WHY), recorded in the line as parallel_raw_why")
    laid_path, video = _resolve_laid_and_video(a)
    share = _resolve_share(a, laid_path)
    existing = [{k: v for k, v in x.items() if k != "line"} for x in read(str(laid_path), str(share), check_bytes=True)] if laid_path.is_file() else []
    if a.cmd in ("replace", "remove") and not laid_path.is_file():
        raise SystemExit(f"{a.cmd}: {laid_path} is missing")

    extra_cites = [str(x.get("owner_word")) for x in existing if x.get("owner_word")]
    if getattr(a, "owner_word", None):
        extra_cites.append(str(a.owner_word))
    choices, _ = _load_owner_choices(laid_path, extra_cites=extra_cites)
    now_mut = (getattr(a, "added", None) or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    carved_whole_records: list[tuple[dict, str, list[list[int]]]] = []
    restored_records: list[tuple[dict, str]] = []
    removed: list[dict] = []

    if a.cmd == "lay":
        first, last = _parse_shot(a, required=True)
        assert first is not None and last is not None
        entry = _build_entry(a, share, first, last, why_no_rebuild)
        if why_partial_lay:
            entry["partial_lay_why"] = why_partial_lay
        if why_parallel_raw:
            entry["parallel_raw_why"] = why_parallel_raw
        new_lines, carved_whole_records = _carve_same_subject_overlaps(
            existing, entry, choices, laid_path, video, getattr(a, "data", None), now_mut, a.cmd
        )
        action_desc = f"laid {entry['load']} on {first}-{last} ({entry['subject']})"
    elif a.cmd == "replace":
        first, last = _parse_shot(a, required=True)
        assert first is not None and last is not None
        target_load = (getattr(a, "replaces", None) or "").strip()
        matched_rows = [
            x for x in existing
            if int(x["first"]) == first and int(x["last"]) == last and x["subject"] == a.subject and (not target_load or x["load"] == target_load)
        ]
        if not matched_rows:
            what = f"load {target_load!r} on {first}-{last} ({a.subject})" if target_load else f"any line on {first}-{last} ({a.subject})"
            raise SystemExit(f"replace: {laid_path} has no line for {what} to replace")
        replaced_loads = [str(x["load"]) for x in matched_rows]
        entry = _build_entry(a, share, first, last, why_no_rebuild, replaces=replaced_loads)
        if why_partial_lay:
            entry["partial_lay_why"] = why_partial_lay
        if why_parallel_raw:
            entry["parallel_raw_why"] = why_parallel_raw
        entry_kept = _is_active_keep(entry, choices)
        candidate_lines: list[dict] = []
        removed_non_keeps: list[dict] = []
        for x in existing:
            if x in matched_rows:
                x_kept = _is_active_keep(x, choices)
                if str(x["load"]) == str(entry["load"]):
                    continue
                if x_kept and not entry_kept:
                    ov = _frame_set(_spans_of_laid(x)) & _frame_set(_spans_of_laid(entry))
                    raise SystemExit(
                        f"replace: {entry['load']} ({entry['owner']}) cannot replace active keep {x['load']} "
                        f"on {entry['subject']} frame(s) {_fmt_spans(_runs_of_frames(ov or _frame_set(_spans_of_laid(x))))}"
                    )
                if not x_kept:
                    removed_non_keeps.append(x)
                    continue
            candidate_lines.append(x)
        if removed_non_keeps:
            restored_records.extend(_restore_carved_by_removed(candidate_lines, removed_non_keeps, laid_path))
        new_lines, carved_whole_records = _carve_same_subject_overlaps(
            candidate_lines, entry, choices, laid_path, video, getattr(a, "data", None), now_mut, a.cmd
        )
        action_desc = f"replaced {', '.join(replaced_loads)} with {entry['load']} on {first}-{last} ({entry['subject']})"
    elif a.cmd == "remove":
        first, last = _parse_shot(a, required=False)
        sub = (getattr(a, "subject", None) or "").strip()
        load = (getattr(a, "load", None) or "").strip()
        if first is None and not sub and not load:
            raise SystemExit("remove: give --load and/or --first --last (or --shot) and --subject")
        kept_lines = []
        for x in existing:
            match_shot = first is None or (int(x["first"]) == first and int(x["last"]) == last)
            match_sub = not sub or x["subject"] == sub
            match_load = not load or x["load"] == load
            if match_shot and match_sub and match_load:
                removed.append(x)
            else:
                kept_lines.append(x)
        if not removed:
            raise SystemExit(f"remove: no line in {laid_path} matched first={first} last={last} subject={sub!r} load={load!r}")
        if not (getattr(a, "by", None) or "").strip():
            raise SystemExit("remove: --by is required: why the line comes out, which is kept in the record of what was removed")
        new_lines = kept_lines
        restored_records.extend(_restore_carved_by_removed(new_lines, removed, laid_path))
        action_desc = "removed " + ", ".join(f"{x['load']} on {x['first']}-{x['last']} ({x['subject']})" for x in removed)
    else:
        raise SystemExit(f"unknown command {a.cmd!r}")

    laid_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = laid_path.with_suffix(".jsonl.tmp")
    tmp_path.write_text("".join(json.dumps(x) + "\n" for x in new_lines))
    read(str(tmp_path), str(share), check_bytes=True)
    if a.cmd in ("lay", "replace") and entry.get("owner") == "unconfirmed":
        assert first is not None and last is not None
        shot_frames = set(range(first, last + 1))
        laid_covered: set[int] = set()
        for r in new_lines:
            if r["subject"] == entry["subject"]:
                laid_covered |= (_frame_set(_spans_of_laid(r)) & shot_frames)
        looks_covered = _not_seen_frames_of_shot(laid_path, video, getattr(a, "data", None), first, last, entry["subject"])
        uncovered = sorted(shot_frames - laid_covered - looks_covered)
        if uncovered and not why_partial_lay:
            tmp_path.unlink(missing_ok=True)
            uncovered_spans = _runs_of_frames(uncovered)
            raise SystemExit(
                f"{a.cmd}: partial lay of {entry['load']} on shot {first}-{last} ({entry['subject']}) leaves uncovered frame(s) "
                f"{_fmt_spans(uncovered_spans)} (neither covered by a laid row on {first}-{last} for {entry['subject']} nor recorded "
                f"`not seen` in shots.json). Lay the remaining piece, record a `not seen` look in shots.json, or pass --allow-partial-lay WHY"
            )
        entry_fset = _frame_set(_spans_of_laid(entry))
        other_overlaps = [
            (r, entry_fset & _frame_set(_spans_of_laid(r)))
            for r in new_lines
            if r["subject"] != entry["subject"] and (entry_fset & _frame_set(_spans_of_laid(r)))
        ]
        if other_overlaps and not why_parallel_raw:
            job_src = _find_load_job_source(laid_path, video, getattr(a, "data", None), str(entry["load"]))
            if job_src:
                for other_r, ov_fset in other_overlaps:
                    holds = _source_holds_other_subject(
                        job_src,
                        share,
                        video,
                        str(other_r["subject"]),
                        ov_fset,
                        laid_path=laid_path,
                        other_load=str(other_r["load"]),
                    )
                    if holds == "cannot_tell":
                        ov_spans = _fmt_spans(_runs_of_frames(ov_fset))
                        print(
                            f"NOTE: {a.cmd}: {entry['load']} ({entry['subject']}) was rendered on cut {Path(job_src).name} "
                            f"which has no readable .json sidecar to verify {other_r['subject']}'s piece on {ov_spans} (source_has_no_record)"
                        )
                    elif holds is False:
                        tmp_path.unlink(missing_ok=True)
                        ov_spans = _fmt_spans(_runs_of_frames(ov_fset))
                        raise SystemExit(
                            f"REFUSED: {a.cmd}: {entry['load']} ({entry['subject']}) was rendered on {Path(job_src).name} "
                            f"which does not hold {other_r['subject']}'s laid piece {other_r['load']} on {ov_spans}. "
                            f"Re-render {entry['load']} as Pass 2 on a cut holding {other_r['load']}, or pass --allow-parallel-raw WHY"
                        )
    sha_before = digest(str(laid_path)) if laid_path.is_file() else None
    sha_after = digest(str(tmp_path))
    if a.cmd == "remove" or carved_whole_records or restored_records:
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        session = (os.environ.get("H3_SESSION") or "").strip() or "not said (no H3_SESSION)"
        with open(removed_path(laid_path), "a") as f:
            for x in removed:
                f.write(json.dumps({"schema": REMOVED_SCHEMA, "removed": x, "by": a.by.strip(), "at": now, "by_session": session,
                                    "laid_sha256_before": sha_before, "laid_sha256_after": sha_after,
                                    **({"no_rebuild": why_no_rebuild} if why_no_rebuild else {})}) + "\n")
            for cw_row, cw_winner, cw_spans in carved_whole_records:
                f.write(json.dumps({"schema": REMOVED_SCHEMA, "removed": cw_row,
                                    "by": f"carved whole by {cw_winner} ({_fmt_spans(cw_spans)})",
                                    "carved_whole_by": cw_winner, "at": now, "by_session": session,
                                    "laid_sha256_before": sha_before, "laid_sha256_after": sha_after,
                                    **({"no_rebuild": why_no_rebuild} if why_no_rebuild else {})}) + "\n")
            for rest_row, from_load in restored_records:
                reason_by = getattr(a, "by", None) or ""
                f.write(json.dumps({"schema": REMOVED_SCHEMA, "restored": rest_row,
                                    "restored_from_carved_whole_by": from_load,
                                    "by": f"restored when {from_load} was removed ({reason_by.strip()})",
                                    "at": now, "by_session": session,
                                    "laid_sha256_before": sha_before, "laid_sha256_after": sha_after,
                                    **({"no_rebuild": why_no_rebuild} if why_no_rebuild else {})}) + "\n")
        if a.cmd == "remove":
            print(f"the record of it: {removed_path(laid_path)}")
    tmp_path.replace(laid_path)
    print(f"{action_desc} in {laid_path} (sha256 {digest(str(laid_path))[:16]}, {len(new_lines)} lines)")
    if why_no_rebuild:
        print(f"no rebuild (--no-rebuild: {why_no_rebuild})")
        return 0
    return _run_autorebuild(a, laid_path, video, share)


def _is_keep(chosen) -> bool:
    """A choice that keeps a render: `delivery_table.KEEP`, or the bare "keep" the lead's terminal took down once."""
    import delivery_table
    said = str(chosen or "").strip().lower()
    return said in {k.lower() for k in delivery_table.KEEP} | {"keep"}


def owner_word(cite: str, load: str, owner: str, cmd: str = "set-owner", laid_path: Path | None = None) -> str:
    """`FILE:LINE`, read and held to being the owner's last choice on its row (by `at`) and a keep (of this load's
    row, for `kept`; of a joined clip's row, a `joined` part in its name, for `kept in a joined clip`). Returns the
    cite as given. Lines are counted as `grep -n` counts them, blank ones included.

    What it cannot see: a later answer the owner gave in another file (a keep cited from the verdicts record is not
    held to answers not yet copied into it), and whether the joined clip it names laid this load."""
    path, sep, num = cite.rpartition(":")
    if not sep or not path or not num.isdigit() or int(num) < 1:
        raise SystemExit(f"{cmd}: --owner-word {cite!r} is not FILE:LINE")
    p_obj = Path(path)
    if not p_obj.is_file() and laid_path is not None:
        vdir = laid_path.resolve().parent
        if path.startswith(f"data/{vdir.name}/"):
            cand_p = vdir / path[len(f"data/{vdir.name}/"):]
            if cand_p.is_file():
                p_obj = cand_p
        elif (vdir / Path(path).name).is_file():
            p_obj = vdir / Path(path).name
    if not p_obj.is_file():
        raise SystemExit(f"{cmd}: --owner-word {cite!r}: {path} is not a file")
    lines = p_obj.read_text().split("\n")
    n = int(num)
    if n > len(lines) or (n == len(lines) and not lines[-1]):
        raise SystemExit(f"{cmd}: --owner-word {cite!r}: {path} has no line {num}")
    if not lines[n - 1].strip():
        raise SystemExit(f"{cmd}: --owner-word {cite!r}: line {num} of {path} is blank")

    def parsed(text: str) -> dict | None:
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            return None
        return d if isinstance(d, dict) else None

    def the_owners(d: dict) -> bool:
        return d.get("who") == "owner" or d.get("schema") == "h3_verdicts/1"

    d = parsed(lines[n - 1])
    if d is None:
        raise SystemExit(f"{cmd}: --owner-word {cite!r}: line {num} of {path} is not a JSON object")
    if not the_owners(d):
        raise SystemExit(f"{cmd}: --owner-word {cite!r} is not the owner's line (who {d.get('who')!r})")
    if not _is_keep(d.get("chosen")):
        raise SystemExit(f"{cmd}: --owner-word {cite!r} is not a keep (what {d.get('what')!r}, chosen {d.get('chosen')!r}): "
                         f"`{owner}` stands on the owner's choice to keep")
    row = str(d.get("row") or "")
    # the owner's last choice on the row is the latest `at`, as delivery_table.last_choice takes it, the file's order
    # only between lines of one time: the verdicts record holds copies of older lines appended late (a line with no
    # `at` is the earliest)
    cited = (str(d.get("at") or ""), n)
    for m, text in enumerate(lines, 1):
        other = parsed(text) if m != n and text.strip() else None
        if other and the_owners(other) and str(other.get("row") or "") == row and str(other.get("chosen") or "").strip() \
                and not _is_keep(other.get("chosen")) and (str(other.get("at") or ""), m) > cited:
            raise SystemExit(f"{cmd}: --owner-word {cite!r} is not the owner's last choice on {row!r}: line {m} of {path} says "
                             f"{other.get('chosen')!r} at {other.get('at')!r}, after this keep's {d.get('at')!r}")
    if owner == "kept" and row != load:
        raise SystemExit(f"{cmd}: --owner-word {cite!r} is the owner's keep of {row!r}, not of {load!r}: `kept` is a keep on "
                         "the render's own row (a keep on a joined clip is `kept in a joined clip`)")
    if owner == "kept in a joined clip" and not re.search(r"(^|_)joined(_|$)", row):
        raise SystemExit(f"{cmd}: --owner-word {cite!r} is a keep of {row!r}, which is not a joined clip's row (no `joined` part in its name)")
    return cite


def set_owner(a: argparse.Namespace) -> int:
    why_no_rebuild: str | None = None
    if a.no_rebuild is not None:
        why_no_rebuild = a.no_rebuild.strip()
        if not why_no_rebuild or why_no_rebuild.startswith("-"):
            raise SystemExit("--no-rebuild requires a non-empty reason (WHY), recorded in the line's history entry")
    if not (a.by or "").strip():
        raise SystemExit("set-owner: --by is required: where the owner's word is, or why the line is not the owner's word")
    if a.owner == "unconfirmed" and a.owner_word:
        raise SystemExit("set-owner: --owner-word cites a keep, and `unconfirmed` is no keep")
    if a.owner != "unconfirmed" and not a.owner_word:
        raise SystemExit(f"set-owner: `{a.owner}` takes --owner-word FILE:LINE, the owner's own line saying the keep")
    laid_path, video = _resolve_laid_and_video(a)
    share = _resolve_share(a, laid_path)
    if not laid_path.is_file():
        raise SystemExit(f"set-owner: {laid_path} is missing")
    first, last = _parse_shot(a, required=False)
    existing = [{k: v for k, v in x.items() if k != "line"} for x in read(str(laid_path), str(share), check_bytes=True)]
    at = [i for i, x in enumerate(existing) if x["load"] == a.load and (first is None or (int(x["first"]), int(x["last"])) == (first, last))
          and (not a.subject or x["subject"] == a.subject)]
    if len(at) != 1:
        raise SystemExit(f"set-owner: {len(at)} lines of {laid_path} are {a.load}" + (f" on {first}-{last}" if first is not None else "")
                         + (f" for {a.subject}" if a.subject else "") + ("; give --shot and --subject" if len(at) > 1 else ""))
    old = existing[at[0]]
    if old["owner"] == a.owner:
        raise SystemExit(f"set-owner: {a.load} on {old['first']}-{old['last']} is `{a.owner}` already")
    cite = owner_word(a.owner_word, a.load, a.owner) if a.owner_word else None
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    session = (a.session or os.environ.get("H3_SESSION") or "").strip() or "not said (no --session and no H3_SESSION)"
    new = {**old, "owner": a.owner, "by": a.by.strip(),
           "history": [*(old.get("history") or []), {"at": now, "by_session": session, "owner_was": old["owner"], "by_was": old["by"],
                                                     "owner": a.owner, "by": a.by.strip(), "owner_word": cite,
                                                     **({"no_rebuild": why_no_rebuild} if why_no_rebuild else {})}]}
    new.pop("owner_word", None)
    if cite:
        new["owner_word"] = cite
    lines = [*existing[:at[0]], new, *existing[at[0] + 1:]]
    tmp_path = laid_path.with_suffix(".jsonl.tmp")
    tmp_path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    read(str(tmp_path), str(share), check_bytes=True)
    tmp_path.replace(laid_path)
    print(f"{a.load} on {old['first']}-{old['last']} ({old['subject']}): owner {old['owner']} -> {a.owner}, line {at[0] + 1} of "
          f"{laid_path} (sha256 {digest(str(laid_path))[:16]}, {len(lines)} lines)")
    if why_no_rebuild:
        print(f"no rebuild (--no-rebuild: {why_no_rebuild})")
        return 0
    return _run_autorebuild(a, laid_path, video, share)


RENDER_LOAD_RE = re.compile(r"^(lead|second|third|fourth)_(?:hfp|fp)_(\d+)([a-z0-9_]*)$")


def _spans_of_laid(x: dict) -> list[list[int]]:
    fr = x.get("frames")
    if isinstance(fr, list) and fr and all(isinstance(s, list) and len(s) == 2 for s in fr):
        return [[int(s[0]), int(s[1])] for s in fr]
    return [[int(x["first"]), int(x["last"])]]


def _frame_set(spans: list[list[int]]) -> set[int]:
    return {f for s0, s1 in spans for f in range(int(s0), int(s1) + 1)}


def _fmt_spans(spans: list[list[int]]) -> str:
    return ",".join(f"{s[0]}-{s[1]}" for s in spans)


def _extract_replaces(obj: dict) -> set[str]:
    rep = obj.get("replaces")
    out: set[str] = set()
    if isinstance(rep, list):
        for item in rep:
            out.update(re.findall(r"\b(?:lead|second|third|fourth)_[a-z0-9_]+\b", str(item)))
            if isinstance(item, str) and item.strip():
                out.add(item.strip().split()[0])
    elif isinstance(rep, str) and rep.strip():
        out.update(re.findall(r"\b(?:lead|second|third|fourth)_[a-z0-9_]+\b", rep))
        out.add(rep.strip().split()[0])
    by = str(obj.get("by") or "")
    for m in re.finditer(r"(?:in place of|replacing|than their earlier keep)\s+([a-z0-9_]+)", by):
        out.add(m.group(1))
    return out


def _laid_effective_at(x: dict, choices: dict[str, dict]) -> str:
    ld = str(x.get("load") or "")
    ch = choices.get(ld)
    if ch and _is_keep(ch["chosen"]) and ch.get("at"):
        return _norm_iso(str(ch["at"]))
    by_times = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z", str(x.get("by") or ""))
    if by_times:
        return max(_norm_iso(t) for t in by_times)
    return _norm_iso(str(x.get("added") or ""))


def _active_removed_rows(rem_rows: list[dict], laid_by_load: dict[str, list[dict]]) -> list[dict]:
    """Filter `rem_rows` from `laid.removed.jsonl` to only active removals (excluding `"restored"` records and any removal
    whose load was later restored or is currently laid in `laid.jsonl`)."""
    active_by_key: dict[tuple[str, str, int, int], list[dict]] = {}
    for r in rem_rows:
        if isinstance(r.get("removed"), dict):
            rm = r["removed"]
            ld = str(rm.get("load") or "")
            if ld in laid_by_load:
                continue
            k = (str(rm.get("subject") or ""), ld, int(rm.get("first", -1)), int(rm.get("last", -1)))
            active_by_key.setdefault(k, []).append(r)
        elif isinstance(r.get("restored"), dict):
            rs = r["restored"]
            k = (str(rs.get("subject") or ""), str(rs.get("load") or ""), int(rs.get("first", -1)), int(rs.get("last", -1)))
            active_by_key.pop(k, None)
    out: list[dict] = []
    for r in rem_rows:
        if isinstance(r.get("removed"), dict):
            rm = r["removed"]
            ld = str(rm.get("load") or "")
            k = (str(rm.get("subject") or ""), ld, int(rm.get("first", -1)), int(rm.get("last", -1)))
            if r in active_by_key.get(k, []):
                out.append(r)
    return out


def audit_cross_subject_provenance(
    laid_rows: list[dict],
    laid_path: Path,
    video: str,
    share: Path | None = None,
    data_arg: str | None = None,
    active_rem_rows: list[dict] | None = None,
) -> dict:
    """Compute frame-level carving and cross-subject cut provenance reports (`carved_frames`,
    `original_restored_over_replacement`, `leaves_as_the_original_not_known`, `stale_pieces`,
    `neither_made_on_the_other`, `source_has_no_record`)."""
    carved_frames: list[dict] = []
    orig_restored_list: list[dict] = []
    not_known_list: list[dict] = []

    for x in laid_rows:
        if x.get("carved_by"):
            carved_frames.append({
                "load": str(x["load"]),
                "subject": str(x["subject"]),
                "shot": f"{x['first']}-{x['last']}",
                "remaining_frames": _spans_of_laid(x),
                "carved_by": x["carved_by"],
            })
        if x.get("original_restored_over_replacement"):
            orig_restored_list.append({
                "load": str(x["load"]),
                "subject": str(x["subject"]),
                "shot": f"{x['first']}-{x['last']}",
                "frames": _spans_of_laid(x),
                "original_restored_over_replacement": x["original_restored_over_replacement"],
            })
        if x.get("leaves_as_the_original_status") == "not_known":
            not_known_list.append({
                "load": str(x["load"]),
                "subject": str(x["subject"]),
                "shot": f"{x['first']}-{x['last']}",
                "frames": _spans_of_laid(x),
            })

    for r in active_rem_rows or []:
        rem = r.get("removed") or {}
        if r.get("carved_whole_by") and rem.get("load"):
            carved_frames.append({
                "load": str(rem["load"]),
                "subject": str(rem.get("subject") or ""),
                "shot": f"{rem.get('first')}-{rem.get('last')}",
                "remaining_frames": [],
                "carved_whole_by": str(r["carved_whole_by"]),
                "carved_by": rem.get("carved_by") or [],
            })

    # Pre-resolve job.source and cut sidecar pieces for every laid row that overlaps another subject
    row_fsets = [_frame_set(_spans_of_laid(r)) for r in laid_rows]
    overlaps_other_sub: set[int] = set()
    for i, r in enumerate(laid_rows):
        for j, o in enumerate(laid_rows):
            if i != j and r["subject"] != o["subject"] and (row_fsets[i] & row_fsets[j]):
                overlaps_other_sub.add(i)
                break

    src_info: dict[int, dict] = {}
    source_has_no_record: list[dict] = []
    seen_no_rec: set[tuple[str, str]] = set()
    for i in sorted(overlaps_other_sub):
        r = laid_rows[i]
        r_load = str(r["load"])
        r_sub = str(r["subject"])
        job_src = _find_load_job_source(laid_path, video, data_arg, r_load)
        if not job_src:
            src_info[i] = {"job_src": None, "is_cut": False, "pieces": None, "no_record": True}
            if (r_load, r_sub) not in seen_no_rec:
                seen_no_rec.add((r_load, r_sub))
                source_has_no_record.append({
                    "load": r_load,
                    "subject": r_sub,
                    "frames": _spans_of_laid(r),
                    "source": None,
                    "why": "no job file or job.source found on disk",
                })
            continue
        is_cut, _sc_path, pieces = _find_cut_sidecar(job_src, share, video, laid_path)
        if is_cut and pieces is None:
            src_info[i] = {"job_src": job_src, "is_cut": True, "pieces": None, "no_record": True}
            if (r_load, r_sub) not in seen_no_rec:
                seen_no_rec.add((r_load, r_sub))
                source_has_no_record.append({
                    "load": r_load,
                    "subject": r_sub,
                    "frames": _spans_of_laid(r),
                    "source": Path(job_src).name,
                    "why": "source cut has no readable .json sidecar pieces",
                })
        else:
            src_info[i] = {"job_src": job_src, "is_cut": is_cut, "pieces": pieces, "no_record": False}

    left_cache: dict[int, tuple[set[int], str]] = {}

    def _row_left(idx: int) -> set[int]:
        if idx not in left_cache:
            left_cache[idx] = _left_by_load_frames(laid_path, video, data_arg, laid_rows[idx])
        s, st = left_cache[idx]
        return s if st == "known" else set()

    stale_pieces: list[dict] = []
    for i, r in enumerate(laid_rows):
        info_r = src_info.get(i)
        if not info_r or info_r["no_record"] or info_r["pieces"] is None:
            continue
        r_pieces = info_r["pieces"]
        r_fset = row_fsets[i]
        for j, o in enumerate(laid_rows):
            if i == j or r["subject"] == o["subject"]:
                continue
            ov_raw = r_fset & row_fsets[j]
            if not ov_raw:
                continue
            ov = (r_fset - _row_left(i)) & (row_fsets[j] - _row_left(j))
            if not ov:
                continue
            o_sub = str(o["subject"])
            o_load = str(o["load"])
            by_old_piece: dict[str, set[int]] = {}
            for p in r_pieces:
                if str(p.get("subject")) != o_sub:
                    continue
                pf, pl = int(p.get("first", -1)), int(p.get("last", -2))
                if pf > pl:
                    continue
                p_ov = set(range(pf, pl + 1)) & ov
                if not p_ov:
                    continue
                p_name = _piece_load_name(p)
                if p_name and p_name != o_load:
                    by_old_piece.setdefault(p_name, set()).update(p_ov)
            for old_name, st_fset in sorted(by_old_piece.items()):
                stale_pieces.append({
                    "load": str(r["load"]),
                    "subject": str(r["subject"]),
                    "other_subject": o_sub,
                    "frames": _runs_of_frames(st_fset),
                    "source_cut": Path(str(info_r["job_src"])).name,
                    "source_piece": old_name,
                    "current_laid_piece": o_load,
                })

    neither_made_on_the_other: list[dict] = []
    for i, r in enumerate(laid_rows):
        for j in range(i + 1, len(laid_rows)):
            o = laid_rows[j]
            if r["subject"] == o["subject"]:
                continue
            ov = row_fsets[i] & row_fsets[j]
            if not ov:
                continue
            info_r = src_info.get(i)
            info_o = src_info.get(j)
            if not info_r or not info_o or info_r["no_record"] or info_o["no_record"]:
                continue
            r_holds_o: set[int] = set()
            if info_r["pieces"] is not None:
                for p in info_r["pieces"]:
                    if str(p.get("subject")) == str(o["subject"]):
                        pf, pl = int(p.get("first", -1)), int(p.get("last", -2))
                        if pf <= pl:
                            r_holds_o.update(range(pf, pl + 1))
            o_holds_r: set[int] = set()
            if info_o["pieces"] is not None:
                for p in info_o["pieces"]:
                    if str(p.get("subject")) == str(r["subject"]):
                        pf, pl = int(p.get("first", -1)), int(p.get("last", -2))
                        if pf <= pl:
                            o_holds_r.update(range(pf, pl + 1))
            neither_fset = ov - (r_holds_o | o_holds_r)
            if neither_fset:
                neither_made_on_the_other.append({
                    "loads": [str(r["load"]), str(o["load"])],
                    "subjects": [str(r["subject"]), str(o["subject"])],
                    "frames": _runs_of_frames(neither_fset),
                    "sources": [
                        Path(str(info_r["job_src"])).name if info_r["job_src"] else None,
                        Path(str(info_o["job_src"])).name if info_o["job_src"] else None,
                    ],
                })

    return {
        "carved_frames": carved_frames,
        "original_restored_over_replacement": orig_restored_list,
        "leaves_as_the_original_not_known": not_known_list,
        "stale_pieces": stale_pieces,
        "neither_made_on_the_other": neither_made_on_the_other,
        "source_has_no_record": source_has_no_record,
    }


def _row_recorded_keep_at(x: dict, laid_path: Path | None = None) -> str:
    """Return the keep timestamp recorded on `x` itself (from `x['owner_word']` if readable on disk, else `x['by']`/`x['owner_looks']`, else `x['added']`)."""
    ow = str(x.get("owner_word") or "").strip()
    if ow:
        p_str, sep, l_str = ow.rpartition(":")
        if sep and p_str and l_str.isdigit():
            cands = [Path(p_str), HERE.parent / p_str]
            if laid_path is not None:
                vdir = laid_path.resolve().parent
                if p_str.startswith(f"data/{vdir.name}/"):
                    cands.insert(0, vdir / p_str[len(f"data/{vdir.name}/"):])
                cands.append(vdir / Path(p_str).name)
            p_file = next((c for c in cands if c.is_file()), None)
            if p_file is not None:
                try:
                    lines = p_file.read_text().splitlines()
                    idx = int(l_str) - 1
                    if 0 <= idx < len(lines) and lines[idx].strip():
                        d = json.loads(lines[idx])
                        if isinstance(d, dict) and d.get("at"):
                            return _norm_iso(str(d["at"]))
                except Exception:
                    pass
    text_blobs = [str(x.get("by") or ""), json.dumps(x.get("owner_looks") or []), json.dumps(x.get("history") or [])]
    by_times = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z", " ".join(text_blobs))
    if by_times:
        return max(_norm_iso(t) for t in by_times)
    return _norm_iso(str(x.get("added") or ""))


def _resolve_load_metadata_for_sync(
    load_name: str,
    laid_path: Path,
    video: str,
    share: Path,
    data_arg: str | None,
    frames_by_row: dict[str, list[list[int]]],
    existing_row: dict | None,
    shots_list: list[dict],
    allow_rglob: bool = False,
) -> dict:
    """Resolve `subject`, `clip_spans`, `shot_first`, `shot_last`, and `rel_file` for `load_name`."""
    search_dirs = _search_data_dirs(laid_path, video, data_arg)
    built_subject: str | None = None
    built_spans: list[list[int]] | None = None
    rel_file: str | None = str(existing_row["file"]) if existing_row and existing_row.get("file") else None

    for d in search_dirs:
        for jdir in sorted(d.glob(f"*_{load_name}_job")):
            for jname in (f"{load_name}.render.built.json", "job.render.json", "job.plan.json", "job.json"):
                jf = jdir / jname
                if not jf.is_file():
                    continue
                try:
                    jd = json.loads(jf.read_text())
                except Exception:
                    continue
                ld_obj = jd.get("load") if isinstance(jd.get("load"), dict) else {}
                if not built_subject and ld_obj.get("subject"):
                    built_subject = str(ld_obj["subject"])
                if built_spans is None and ld_obj.get("first") is not None and ld_obj.get("frames"):
                    f0 = int(ld_obj["first"])
                    built_spans = [[f0, f0 + int(ld_obj["frames"]) - 1]]
                if rel_file is None and jd.get("output_folder"):
                    out_f = str(jd["output_folder"])
                    for cand_name in (f"{load_name}_00001.mp4", f"{load_name}.mp4"):
                        cand_abs = Path(out_f) / cand_name
                        if cand_abs.is_file():
                            try:
                                rel_file = str(cand_abs.resolve().relative_to(share.resolve()))
                            except Exception:
                                if "/Video/" in str(cand_abs):
                                    rel_file = str(cand_abs)[str(cand_abs).index("Video/"):]
                            break
                        if "/Video/" in out_f:
                            rel_cand = out_f[out_f.index("Video/"):].rstrip("/") + "/" + cand_name
                            if (share / rel_cand).is_file():
                                rel_file = rel_cand
                                break
            qp = jdir / f"{load_name}.render.queued.json"
            if rel_file is None and qp.is_file():
                try:
                    qdoc = json.loads(qp.read_text())
                    qfiles = qdoc.get("files") or []
                    for qf in qfiles:
                        qf_str = str(qf).lstrip("/")
                        if (share / qf_str).is_file():
                            rel_file = qf_str
                            break
                except Exception:
                    pass

    subject = (
        (str(existing_row["subject"]) if existing_row and existing_row.get("subject") else None)
        or built_subject
    )
    if not subject:
        m = RENDER_LOAD_RE.match(load_name) or re.match(r"^(lead|second|third|fourth)_(?:hfp|fp|hf|hp|g)_", load_name)
        if m:
            subject = m.group(1)

    if existing_row is not None:
        base_fset = _frame_set(_spans_of_laid(existing_row))
        carved_fset = _frame_set([s for cb in (existing_row.get("carved_by") or []) for s in (cb.get("frames") or [])])
        clip_spans = _runs_of_frames(base_fset | carved_fset)
    else:
        clip_spans = built_spans or frames_by_row.get(load_name)

    if rel_file is None and allow_rglob and share.is_dir():
        v_root = share / "Video" / video if (share / "Video" / video).is_dir() else share
        for pat in (f"{load_name}_00001.mp4", f"{load_name}.mp4"):
            found = sorted(v_root.rglob(pat))
            if not found and v_root != share:
                found = sorted(share.rglob(pat))
            if found:
                rel_file = str(found[0].resolve().relative_to(share.resolve()))
                break

    shot_first: int | None = int(existing_row["first"]) if existing_row and existing_row.get("first") is not None else None
    shot_last: int | None = int(existing_row["last"]) if existing_row and existing_row.get("last") is not None else None
    if (shot_first is None or shot_last is None) and clip_spans:
        cf0, cf1 = int(clip_spans[0][0]), int(clip_spans[-1][1])
        clip_fset = _frame_set(clip_spans)
        best_shot: tuple[int, int] | None = None
        best_ov = 0
        for s in shots_list:
            sf, sl = int(s["first"]), int(s["last"])
            if sf <= cf0 and cf1 <= sl:
                best_shot = (sf, sl)
                break
            ov_len = len(clip_fset & set(range(sf, sl + 1)))
            if ov_len > best_ov:
                best_ov = ov_len
                best_shot = (sf, sl)
        if best_shot is not None:
            shot_first, shot_last = best_shot
        else:
            shot_first, shot_last = cf0, cf1

    all_not_seen = False
    if existing_row is None and clip_spans and subject and shot_first is not None and shot_last is not None:
        clip_fset = _frame_set(clip_spans)
        not_seen_fset: set[int] = set()
        for s in shots_list:
            sf, sl = int(s["first"]), int(s["last"])
            if clip_fset & set(range(sf, sl + 1)):
                not_seen_fset |= _not_seen_frames_of_shot(laid_path, video, data_arg, sf, sl, subject)
        if not not_seen_fset:
            not_seen_fset = _not_seen_frames_of_shot(laid_path, video, data_arg, shot_first, shot_last, subject)
        if not_seen_fset:
            rem_fset = clip_fset - not_seen_fset
            clip_spans = _runs_of_frames(rem_fset)
            if not clip_spans:
                all_not_seen = True

    return {
        "subject": subject,
        "clip_spans": clip_spans,
        "shot_first": shot_first,
        "shot_last": shot_last,
        "rel_file": rel_file,
        "all_not_seen": all_not_seen,
    }


def sync(a: argparse.Namespace) -> int:
    """Synchronize `laid.jsonl` with all standing owner keep clicks in `owner_answers.jsonl` and `verdicts.jsonl`
    in chronological order (`(at, file_idx, line_num)`). Each kept clip takes every frame in its clip for its subject,
    carving any earlier row of the same subject."""
    laid_path, video = _resolve_laid_and_video(a)
    share = _resolve_share(a, laid_path)
    existing = (
        [{k: v for k, v in x.items() if k != "line"} for x in read(str(laid_path), str(share), check_bytes=True)]
        if laid_path.is_file()
        else []
    )
    rem_file = removed_path(laid_path)
    rem_rows: list[dict] = []
    if rem_file.is_file():
        for text in rem_file.read_text().splitlines():
            if text.strip():
                try:
                    rem_rows.append(json.loads(text))
                except Exception:
                    continue

    existing_by_load: dict[str, dict] = {str(x["load"]): x for x in existing}
    laid_by_load_list: dict[str, list[dict]] = {}
    for x in existing:
        laid_by_load_list.setdefault(str(x["load"]), []).append(x)
    active_rem_rows = _active_removed_rows(rem_rows, laid_by_load_list)

    explicit_rem_at: dict[str, str] = {}
    carved_whole_by_load: dict[str, dict] = {}
    for r in active_rem_rows:
        rm = r.get("removed")
        if not isinstance(rm, dict) or not rm.get("load"):
            continue
        ld = str(rm["load"])
        if r.get("carved_whole_by"):
            carved_whole_by_load[ld] = rm
        else:
            explicit_rem_at[ld] = max(explicit_rem_at.get(ld, ""), _norm_iso(str(r.get("at") or "")))

    extra_cites = [str(x.get("owner_word")) for x in existing if x.get("owner_word")]
    choices, frames_by_row = _load_owner_choices(
        laid_path,
        answers_arg=getattr(a, "answers", None),
        verdicts_arg=getattr(a, "verdicts", None),
        extra_cites=extra_cites,
    )

    shots_list: list[dict] = []
    for d in _search_data_dirs(laid_path, video, getattr(a, "data", None)):
        for cand in (d / "shots.json", d / video / "shots.json"):
            if cand.is_file():
                try:
                    sdoc = json.loads(cand.read_text())
                    if isinstance(sdoc.get("shots"), list):
                        shots_list = sdoc["shots"]
                        break
                except Exception:
                    continue
        if shots_list:
            break

    unlaid_keeps: list[dict] = []
    meta_by_load: dict[str, dict] = {}
    click_events: list[tuple[tuple[str, int, int], str, str, list[list[int]], str, dict]] = []

    for row, info in sorted(choices.items(), key=lambda kv: kv[1]["key"]):
        if not _is_keep(info["chosen"]):
            continue
        if re.search(r"(^|_)joined(_|$)", row):
            unlaid_keeps.append({
                "load": row,
                "at": info["at"],
                "cite": info["cite"],
                "why": "joined clip (not a single-subject render load)",
            })
            continue
        if row not in existing_by_load and row in explicit_rem_at and explicit_rem_at[row] >= _norm_iso(info["at"]):
            unlaid_keeps.append({
                "load": row,
                "at": info["at"],
                "cite": info["cite"],
                "why": f"explicitly removed at {explicit_rem_at[row]}",
            })
            continue
        if row not in existing_by_load:
            rep_by_laid: str | None = None
            info_at_norm = _norm_iso(info["at"])
            for x in existing:
                if row in _extract_replaces(x) and _is_active_keep(x, choices):
                    x_at = _laid_effective_at(x, choices)
                    if not info_at_norm or not x_at or x_at >= info_at_norm:
                        rep_by_laid = f"{x['load']} ({x_at})"
                        break
            if rep_by_laid:
                unlaid_keeps.append({
                    "load": row,
                    "at": info["at"],
                    "cite": info["cite"],
                    "why": f"replaced by {rep_by_laid}",
                })
                continue
        known_row = existing_by_load.get(row) or carved_whole_by_load.get(row)
        meta = _resolve_load_metadata_for_sync(
            row, laid_path, video, share, getattr(a, "data", None), frames_by_row, known_row, shots_list
        )
        meta_by_load[row] = meta
        if meta.get("all_not_seen"):
            unlaid_keeps.append({
                "load": row,
                "at": info["at"],
                "cite": info["cite"],
                "why": "all frames marked not seen in shots.json",
            })
            continue
        if not meta["subject"] or not meta["clip_spans"]:
            unlaid_keeps.append({
                "load": row,
                "at": info["at"],
                "cite": info["cite"],
                "why": "not a single-subject render load",
            })
            continue
        click_events.append((info["key"], row, str(meta["subject"]), meta["clip_spans"], "click", info))

    click_loads = {ev[1] for ev in click_events}
    other_events: list[tuple[tuple[str, int, int], str, str, list[list[int]], str, dict]] = []
    for idx, x in enumerate(existing):
        ld = str(x["load"])
        if ld in click_loads:
            continue
        is_k = _is_active_keep(x, choices)
        eff_at = _laid_effective_at(x, choices) if is_k else ""
        base_fset = _frame_set(_spans_of_laid(x))
        carved_fset = _frame_set([s for cb in (x.get("carved_by") or []) for s in (cb.get("frames") or [])])
        x_spans = _runs_of_frames(base_fset | carved_fset)
        other_events.append(((eff_at, 9, idx), ld, str(x["subject"]), x_spans, "laid_existing", x))

    all_events = sorted([*click_events, *other_events], key=lambda ev: ev[0])

    # Iteratively filter out any click event that would survive to the end of the timeline but has no render file on disk
    while True:
        sim_fsets: dict[str, tuple[str, set[int], str, dict]] = {}
        for _k, ld, sub, c_spans, kind, obj in all_events:
            fset = _frame_set(c_spans)
            for old_ld, (old_sub, old_fset, _okind, _oobj) in list(sim_fsets.items()):
                if old_sub == sub:
                    old_fset -= fset
                    if not old_fset:
                        del sim_fsets[old_ld]
            sim_fsets[ld] = (sub, set(fset), kind, obj)
        bad_loads: set[str] = set()
        for ld, (_sub, rem_fset, kind, obj) in sim_fsets.items():
            if not rem_fset or ld in existing_by_load:
                continue
            meta = meta_by_load.get(ld) or {}
            rel_f = meta.get("rel_file")
            if not rel_f:
                known_row = existing_by_load.get(ld) or carved_whole_by_load.get(ld)
                meta = _resolve_load_metadata_for_sync(
                    ld, laid_path, video, share, getattr(a, "data", None), frames_by_row, known_row, shots_list, allow_rglob=True
                )
                meta_by_load[ld] = meta
                rel_f = meta.get("rel_file")
            if not rel_f or not (share / rel_f).is_file():
                bad_loads.add(ld)
                why_msg = f"render file not found on disk under {share}"
                unlaid_keeps.append({
                    "load": ld,
                    "at": obj.get("at", ""),
                    "cite": obj.get("cite", ""),
                    "why": why_msg,
                })
                print(f"UNSYNCED_KEEP: {ld} ({obj.get('cite', '')} at {obj.get('at', '')}): {why_msg}")
        if not bad_loads:
            break
        all_events = [ev for ev in all_events if ev[1] not in bad_loads]

    # Run the full chronological carving pass
    now_sync = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    active_by_load: dict[str, dict] = {}
    carved_whole_by_winner: dict[str, tuple[str, list[list[int]], list[dict]]] = {}

    for key, ld, sub, c_spans, kind, obj in all_events:
        e_fset = _frame_set(c_spans)
        w_at = key[0] or now_sync
        is_keep_ev = True if kind == "click" else _is_active_keep(obj, choices)
        carved_from_list: list[tuple[str, list[list[int]], set[int]]] = []
        carved_whole_loads: list[str] = []
        for old_ld, st in list(active_by_load.items()):
            if st["subject"] != sub:
                continue
            ov = st["fset"] & e_fset
            if not ov:
                continue
            carved_spans = _runs_of_frames(ov)
            _merge_carved_by(
                st,
                ld,
                carved_spans,
                w_at,
                f"carved by newer keep {ld} ({w_at})",
            )
            if st.get("is_keep", True) and is_keep_ev:
                carved_from_list.append((old_ld, st["clip_spans"], set(ov)))
                for prev_old, prev_spans, prev_ov in st["carved_from"]:
                    if prev_ov & ov:
                        carved_from_list.append((prev_old, prev_spans, prev_ov & ov))
            st["fset"] -= ov
            if not st["fset"]:
                carved_whole_by_winner[old_ld] = (ld, carved_spans, list(st.get("carved_by") or []))
                carved_whole_loads.append(old_ld)
                for prev_cw in st.get("carved_whole_loads") or []:
                    if prev_cw not in carved_whole_loads:
                        carved_whole_loads.append(prev_cw)
                del active_by_load[old_ld]
        active_by_load[ld] = {
            "load": ld,
            "subject": sub,
            "clip_spans": c_spans,
            "fset": set(e_fset),
            "carved_by": [],
            "carved_from": carved_from_list,
            "carved_whole_loads": carved_whole_loads,
            "kind": kind,
            "is_keep": is_keep_ev,
            "obj": obj,
            "key": key,
        }

    # Check original_restored_over_replacement on surviving rows that carved an older row
    left_memo: dict[str, tuple[set[int], str]] = {}

    def _get_left_for_sync(load_n: str, sub_n: str, spans_n: list[list[int]]) -> tuple[set[int], str]:
        if load_n not in left_memo:
            f0, f1 = int(spans_n[0][0]), int(spans_n[-1][1])
            left_memo[load_n] = _left_by_load_frames(
                laid_path, video, getattr(a, "data", None), {"load": load_n, "subject": sub_n, "first": f0, "last": f1}
            )
        return left_memo[load_n]

    orig_restored_report: list[dict] = []
    for ld, st in active_by_load.items():
        if not st["carved_from"]:
            continue
        sub = st["subject"]
        # Check each older load on frames currently owned by ld
        by_older: dict[str, tuple[list[list[int]], set[int]]] = {}
        for old_ld, old_spans, ov_set in st["carved_from"]:
            ov_now = ov_set & st["fset"]
            if not ov_now:
                continue
            prev_sp, prev_f = by_older.get(old_ld, (old_spans, set()))
            by_older[old_ld] = (prev_sp, prev_f | ov_now)
        if not by_older:
            continue
        w_left, w_status = _get_left_for_sync(ld, sub, st["clip_spans"])
        if w_status == "not_known":
            st["leaves_as_the_original_status"] = "not_known"
        if not w_left:
            continue
        for old_ld, (old_spans, ov_now) in sorted(by_older.items()):
            if not (ov_now & w_left):
                continue
            l_left, _l_status = _get_left_for_sync(old_ld, sub, old_spans)
            restored_orig = (ov_now & w_left) - l_left
            if restored_orig:
                r_spans = _runs_of_frames(restored_orig)
                _merge_original_restored(
                    st,
                    old_ld,
                    r_spans,
                    "newer keep leaves frame(s) as original where older keep replaced subject",
                )
                orig_restored_report.append({
                    "load": ld,
                    "older_load": old_ld,
                    "subject": sub,
                    "frames": r_spans,
                })

    # Construct new_lines preserving existing order for surviving rows and appending newly synced keeps
    session = (os.environ.get("H3_SESSION") or "").strip() or "laid.py sync"
    new_lines: list[dict] = []
    carved_whole_records: list[tuple[dict, str, list[list[int]]]] = []
    restored_records: list[tuple[dict, str]] = []
    changes_made: list[dict] = []

    for old_row in existing:
        ld = str(old_row["load"])
        if ld not in active_by_load:
            cw_info = carved_whole_by_winner.get(ld)
            winner_ld = cw_info[0] if cw_info else "newer_keep"
            cw_spans = cw_info[1] if cw_info else _spans_of_laid(old_row)
            cw_cby = cw_info[2] if cw_info else []
            rem_copy = json.loads(json.dumps(old_row))
            if cw_cby:
                rem_copy["carved_by"] = cw_cby
            carved_whole_records.append((rem_copy, winner_ld, cw_spans))
            changes_made.append({
                "action": "carved_whole",
                "load": ld,
                "subject": str(old_row["subject"]),
                "was_frames": _spans_of_laid(old_row),
                "by_load": winner_ld,
            })
            continue

        st = active_by_load[ld]
        new_row = json.loads(json.dumps(old_row))
        new_spans = _runs_of_frames(st["fset"])
        old_spans = _spans_of_laid(old_row)

        if st["kind"] == "click":
            info = st["obj"]
            rec_at = _row_recorded_keep_at(old_row, laid_path)
            old_ow = str(old_row.get("owner_word") or "")
            old_ow_base = f"{Path(old_ow.rpartition(':')[0]).name}:{old_ow.rpartition(':')[2]}" if ":" in old_ow else old_ow
            same_cite = bool(old_ow and (old_ow == info.get("cite_path") or old_ow_base == info.get("cite")))
            is_newer_rekeep = bool(
                info.get("at")
                and rec_at
                and (_norm_iso(info["at"]) > rec_at if old_row.get("owner_word") else str(info["at"])[:16] > rec_at[:16])
                and not same_cite
                and (old_row.get("owner") != "kept" or new_spans != old_spans)
            )
            if old_row.get("owner") == "unconfirmed" or is_newer_rekeep:
                new_by = f"the owner's keep on the review page ({info['cite']}, {info['at']})"
                new_row["history"] = [
                    *(old_row.get("history") or []),
                    {
                        "at": now_sync,
                        "by_session": session,
                        "owner_was": old_row.get("owner"),
                        "by_was": old_row.get("by"),
                        "owner": "kept",
                        "by": new_by,
                        "owner_word": info["cite_path"],
                    },
                ]
                new_row["owner"] = "kept"
                new_row["by"] = new_by
                new_row["owner_word"] = info["cite_path"]

        for cw_ld in st.get("carved_whole_loads") or []:
            cw_old = existing_by_load.get(cw_ld)
            if cw_old and (int(cw_old["first"]), int(cw_old["last"])) == (int(new_row["first"]), int(new_row["last"])):
                _merge_replaces(new_row, [cw_ld, *_replaces_list(cw_old)])

        new_row["frames"] = new_spans
        if st.get("carved_by"):
            old_cby_map = {
                (str(c.get("load")), json.dumps(c.get("frames"))): c
                for c in (old_row.get("carved_by") or [])
            }
            merged_cby: list[dict] = []
            for cb in st["carved_by"]:
                k_cb = (str(cb.get("load")), json.dumps(cb.get("frames")))
                merged_cby.append(old_cby_map.get(k_cb, cb))
            new_row["carved_by"] = merged_cby
        else:
            new_row.pop("carved_by", None)

        if st.get("original_restored_over_replacement"):
            new_row["original_restored_over_replacement"] = st["original_restored_over_replacement"]
        else:
            new_row.pop("original_restored_over_replacement", None)

        if new_spans != old_spans or new_row.get("owner") != old_row.get("owner") or new_row.get("owner_word") != old_row.get("owner_word"):
            changes_made.append({
                "action": "updated",
                "load": ld,
                "subject": str(new_row["subject"]),
                "old_frames": old_spans,
                "new_frames": new_spans,
            })
        new_lines.append(new_row)

    # Add newly laid rows from active_by_load that were not in existing
    for ld, st in sorted(active_by_load.items(), key=lambda kv: kv[1]["key"]):
        if ld in existing_by_load:
            continue
        info = st["obj"]
        meta = meta_by_load[ld]
        rel_file = str(meta["rel_file"])
        full_file = share / rel_file
        z = zoomed(full_file)
        cite_verified = owner_word(info["cite_path"], ld, "kept", cmd="sync", laid_path=laid_path)
        new_spans = _runs_of_frames(st["fset"])
        new_entry: dict = {
            "schema": SCHEMA,
            "first": int(meta["shot_first"]),
            "last": int(meta["shot_last"]),
            "subject": str(st["subject"]),
            "load": ld,
            "file": rel_file,
            "sha256": sha256(full_file),
            **({"zoomed_sha256": sha256(z)} if z.is_file() else {}),
            "owner": "kept",
            "by": f"the owner's keep on the review page ({info['cite']}, {info['at']})",
            "owner_word": cite_verified,
            "frames": new_spans,
            "added": str(info.get("at") or now_sync),
            "added_by": session,
        }
        if ld in carved_whole_by_load and carved_whole_by_load[ld].get("replaces"):
            _merge_replaces(new_entry, _replaces_list(carved_whole_by_load[ld]))
        for cw_ld in st.get("carved_whole_loads") or []:
            cw_old = existing_by_load.get(cw_ld) or carved_whole_by_load.get(cw_ld)
            cw_meta = meta_by_load.get(cw_ld)
            if cw_old and (int(cw_old["first"]), int(cw_old["last"])) == (int(new_entry["first"]), int(new_entry["last"])):
                _merge_replaces(new_entry, [cw_ld, *_replaces_list(cw_old)])
            elif cw_meta and (cw_meta.get("shot_first"), cw_meta.get("shot_last")) == (int(new_entry["first"]), int(new_entry["last"])):
                _merge_replaces(new_entry, [cw_ld])
        if st.get("carved_by"):
            new_entry["carved_by"] = st["carved_by"]
        if st.get("original_restored_over_replacement"):
            new_entry["original_restored_over_replacement"] = st["original_restored_over_replacement"]
        if ld in carved_whole_by_load:
            restored_records.append((json.loads(json.dumps(new_entry)), str(carved_whole_by_load[ld].get("load") or ld)))
        new_lines.append(new_entry)
        changes_made.append({
            "action": "laid",
            "load": ld,
            "subject": str(st["subject"]),
            "shot": f"{meta['shot_first']}-{meta['shot_last']}",
            "frames": new_spans,
        })

    if new_lines != existing or carved_whole_records or restored_records:
        laid_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = laid_path.with_suffix(".jsonl.tmp")
        tmp_path.write_text("".join(json.dumps(x) + "\n" for x in new_lines))
        read(str(tmp_path), str(share), check_bytes=True)
        sha_before = digest(str(laid_path)) if laid_path.is_file() else None
        sha_after = digest(str(tmp_path))
        if carved_whole_records or restored_records:
            with open(removed_path(laid_path), "a") as f:
                for cw_row, cw_winner, cw_spans in carved_whole_records:
                    f.write(json.dumps({
                        "schema": REMOVED_SCHEMA,
                        "removed": cw_row,
                        "by": f"carved whole by {cw_winner} ({_fmt_spans(cw_spans)})",
                        "carved_whole_by": cw_winner,
                        "at": now_sync,
                        "by_session": session,
                        "laid_sha256_before": sha_before,
                        "laid_sha256_after": sha_after,
                        "no_rebuild": "synced by laid.py sync",
                    }) + "\n")
                for rest_row, from_load in restored_records:
                    f.write(json.dumps({
                        "schema": REMOVED_SCHEMA,
                        "restored": rest_row,
                        "restored_from_carved_whole_by": from_load,
                        "by": f"restored by newer keep click on {rest_row['load']}",
                        "at": now_sync,
                        "by_session": session,
                        "laid_sha256_before": sha_before,
                        "laid_sha256_after": sha_after,
                        "no_rebuild": "synced by laid.py sync",
                    }) + "\n")
        tmp_path.replace(laid_path)
        print(f"sync: updated {laid_path} ({len(changes_made)} change(s), {len(new_lines)} lines, sha256 {sha_after[:16]})")
        for chg in changes_made:
            if chg["action"] == "laid":
                print(f"  + laid {chg['load']} ({chg['subject']}) on shot {chg['shot']} frames {_fmt_spans(chg['frames'])}")
            elif chg["action"] == "carved_whole":
                print(f"  - carved whole {chg['load']} ({chg['subject']}) was {_fmt_spans(chg['was_frames'])} by {chg['by_load']}")
            else:
                print(f"  ~ {chg['load']} ({chg['subject']}) frames {_fmt_spans(chg['old_frames'])} -> {_fmt_spans(chg['new_frames'])}")
    else:
        print(f"sync: {laid_path} already in sync ({len(new_lines)} lines)")

    if orig_restored_report:
        print(f"=== Original restored over older replacement ({len(orig_restored_report)}) ===")
        for orr in orig_restored_report:
            print(f"  {orr['load']} ({orr['subject']}) leaves {_fmt_spans(orr['frames'])} as original where {orr['older_load']} replaced {orr['subject']}")

    if getattr(a, "json", None):
        Path(a.json).write_text(json.dumps({
            "laid_path": str(laid_path),
            "changes": changes_made,
            "original_restored_over_replacement": orig_restored_report,
            "unlaid_historical_or_non_subject_keeps": unlaid_keeps,
            "lines": len(new_lines),
        }, indent=2) + "\n")
    return 0


def keeps(a: argparse.Namespace) -> int:
    laid_path, video = _resolve_laid_and_video(a)
    if not laid_path.is_file():
        raise SystemExit(f"keeps: {laid_path} is missing")
    laid_rows = read(str(laid_path), "", check_bytes=False)
    vdir = laid_path.resolve().parent
    rem_path = Path(a.removed) if getattr(a, "removed", None) else removed_path(laid_path)
    rem_rows: list[dict] = []
    if rem_path.is_file():
        for text in rem_path.read_text().splitlines():
            if text.strip():
                rem_rows.append(json.loads(text))

    extra_cites = [str(x.get("owner_word")) for x in laid_rows if x.get("owner_word")]
    choices, frames_by_row = _load_owner_choices(
        laid_path,
        answers_arg=getattr(a, "answers", None),
        verdicts_arg=getattr(a, "verdicts", None),
        extra_cites=extra_cites,
    )
    subject_by_row: dict[str, str] = {}
    for r in rem_rows:
        rem = r.get("removed") or r.get("restored") or {}
        if rem.get("load"):
            frames_by_row.setdefault(str(rem["load"]), _spans_of_laid(rem))
            if rem.get("subject"):
                subject_by_row[str(rem["load"])] = str(rem["subject"])
    for x in laid_rows:
        frames_by_row[str(x["load"])] = _spans_of_laid(x)
        subject_by_row[str(x["load"])] = str(x["subject"])

    by_shot: dict[str, list[dict]] = {}
    by_shot_sub: dict[tuple[str, str], list[dict]] = {}
    laid_by_load: dict[str, list[dict]] = {}
    for x in laid_rows:
        shot = f"{x['first']}-{x['last']}"
        by_shot.setdefault(shot, []).append(x)
        by_shot_sub.setdefault((shot, str(x["subject"])), []).append(x)
        laid_by_load.setdefault(str(x["load"]), []).append(x)

    active_rem_rows = _active_removed_rows(rem_rows, laid_by_load)

    multi_sub_shots = {s for s, rows in by_shot.items() if len({r["subject"] for r in rows}) >= 2}
    split_span_shots: set[tuple[str, str]] = set()
    for k, rows in by_shot_sub.items():
        if len(rows) >= 2:
            sets = [_frame_set(_spans_of_laid(r)) for r in rows]
            if all(not (sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))):
                split_span_shots.add(k)

    laid_report: list[dict] = []
    wrong_laid: list[dict] = []
    for x in laid_rows:
        ld = str(x["load"])
        shot = f"{x['first']}-{x['last']}"
        sub = str(x["subject"])
        spans = _spans_of_laid(x)
        ch = choices.get(ld)
        eff_at = _laid_effective_at(x, choices)
        tags: list[str] = []
        if shot in multi_sub_shots:
            tags.append("multi-subject")
        if (shot, sub) in split_span_shots:
            tags.append("split-span")
        if x.get("carved_by"):
            tags.append("carved")
        if x.get("original_restored_over_replacement"):
            tags.append("original-restored-over-replacement")
        is_wrong = False
        if ch is not None and not _is_keep(ch["chosen"]):
            ch_low = ch["chosen"].strip().lower()
            if x["owner"] == "kept":
                is_wrong = True
            elif x["owner"] == "kept in a joined clip":
                if ch_low != "not sure" or (ch["at"] and eff_at and _norm_iso(ch["at"]) > eff_at) or (ch["at"] and x.get("added") and _norm_iso(ch["at"]) > _norm_iso(str(x.get("added")))):
                    is_wrong = True
            else:
                if ch_low != "not sure":
                    is_wrong = True
        if ch is not None:
            last_choice_str = f"{ch['chosen']} ({ch['at'] or 'no-at'}, {ch['cite']})"
        elif x["owner"] == "kept in a joined clip":
            last_choice_str = f"joined-clip keep ({eff_at})"
        elif x["owner"] == "kept":
            last_choice_str = f"whole-video/terminal keep ({eff_at})"
        else:
            last_choice_str = "none"
        status = f"LAID_MARKED_WRONG ({last_choice_str})" if is_wrong else ("ok" + (f" [{', '.join(tags)}]" if tags else ""))
        rec = {
            "line": x["line"],
            "shot": shot,
            "frames": spans,
            "subject": sub,
            "load": ld,
            "owner": x["owner"],
            "last_owner_choice": last_choice_str,
            "tags": tags,
            "status": status,
        }
        if x.get("carved_by"):
            rec["carved_by"] = x["carved_by"]
        if x.get("original_restored_over_replacement"):
            rec["original_restored_over_replacement"] = x["original_restored_over_replacement"]
        if x.get("leaves_as_the_original_status"):
            rec["leaves_as_the_original_status"] = x["leaves_as_the_original_status"]
        laid_report.append(rec)
        if is_wrong:
            wrong_laid.append(rec)

    kept_loads: list[tuple[str, dict]] = []
    for row, info in sorted(choices.items()):
        if re.search(r"(^|_)joined(_|$)", row):
            continue
        if not _is_keep(info["chosen"]):
            continue
        m = RENDER_LOAD_RE.match(row)
        if not m and row not in subject_by_row:
            continue
        kept_loads.append((row, info))

    shots_list: list[dict] = []
    for d in _search_data_dirs(laid_path, video, getattr(a, "data", None)):
        for cand in (d / "shots.json", d / video / "shots.json"):
            if cand.is_file():
                try:
                    sdoc = json.loads(cand.read_text())
                    if isinstance(sdoc.get("shots"), list):
                        shots_list = sdoc["shots"]
                        break
                except Exception:
                    continue
        if shots_list:
            break

    superseded_keeps: list[dict] = []
    missing_keeps: list[dict] = []
    for row, info in kept_loads:
        if row in laid_by_load:
            continue
        m = RENDER_LOAD_RE.match(row)
        sub = m.group(1) if m else subject_by_row.get(row, "")
        r_first = int(m.group(2)) if m else 0
        spans = frames_by_row.get(row) or ([[r_first, r_first]] if r_first else [])
        fset = _frame_set(spans) if spans else set()
        info_at_norm = _norm_iso(info["at"])
        sup_by: str | None = None
        for x in laid_rows:
            if row in _extract_replaces(x) and _is_active_keep(x, choices):
                x_at = _laid_effective_at(x, choices)
                if not info_at_norm or not x_at or x_at >= info_at_norm:
                    sup_by = f"{x['load']} ({x_at})"
                    break
        if not sup_by and fset:
            not_seen_fset: set[int] = set()
            for s in shots_list:
                sf, sl = int(s["first"]), int(s["last"])
                if fset & set(range(sf, sl + 1)):
                    not_seen_fset |= _not_seen_frames_of_shot(laid_path, video, getattr(a, "data", None), sf, sl, sub)
            seen_fset = fset - not_seen_fset
            if not seen_fset and not_seen_fset:
                sup_by = "shots.json (not seen)"
            else:
                # Check across all laid rows of the same subject (across all shots): only a newer keep timestamp supersedes
                covering_newer: list[str] = []
                covered_fset: set[int] = set()
                for x in laid_rows:
                    if str(x["subject"]) != sub:
                        continue
                    ov = seen_fset & _frame_set(_spans_of_laid(x))
                    if ov:
                        x_at = _laid_effective_at(x, choices)
                        if not info_at_norm or (x_at and x_at >= info_at_norm):
                            covered_fset |= ov
                            covering_newer.append(f"{x['load']} ({x_at})")
                if covering_newer and covered_fset == seen_fset:
                    sup_by = ", ".join(covering_newer)
        if not sup_by:
            for r in active_rem_rows:
                rem = r.get("removed") or {}
                if str(rem.get("subject") or "") != sub:
                    continue
                rem_at = _norm_iso(str(r.get("at") or ""))
                rem_eff = _laid_effective_at(rem, choices)
                if r.get("carved_whole_by") and row == rem.get("load"):
                    sup_by = f"{r['carved_whole_by']} (carved whole {rem_at})"
                    break
                if rem_at and info_at_norm and rem_at < info_at_norm:
                    continue
                if row == rem.get("load") or row in _extract_replaces(rem) or (
                    fset and (fset & _frame_set(_spans_of_laid(rem))) and (not info_at_norm or rem_eff >= info_at_norm)
                ):
                    sup_by = f"{rem.get('load')} (removed {rem_at})"
                    break
        entry = {
            "load": row,
            "subject": sub,
            "frames": spans,
            "keep_at": info["at"],
            "cite": info["cite"],
            "superseded_by": sup_by,
        }
        if sup_by:
            superseded_keeps.append(entry)
        else:
            missing_keeps.append(entry)

    # Removed rows in laid.removed.jsonl (active removals only) whose (subject, frames) have no current laid row in laid.jsonl
    unreplaced_removals: list[dict] = []
    seen_rem_keys: set[tuple[str, str]] = set()
    for r in reversed(active_rem_rows):
        rem = r.get("removed") or {}
        rem_load = str(rem.get("load") or "")
        rem_sub = str(rem.get("subject") or "")
        rem_spans = _spans_of_laid(rem) if rem.get("first") is not None else []
        rem_fset = _frame_set(rem_spans) if rem_spans else set()
        rkey = (rem_sub, _fmt_spans(rem_spans))
        if not rem_sub or not rem_fset or rkey in seen_rem_keys:
            continue
        seen_rem_keys.add(rkey)
        covered_now = any(
            str(x["subject"]) == rem_sub and bool(rem_fset & _frame_set(_spans_of_laid(x)))
            for x in laid_rows
        )
        if not covered_now:
            unreplaced_removals.append({
                "shot": f"{rem.get('first')}-{rem.get('last')}",
                "subject": rem_sub,
                "frames": rem_spans,
                "removed_load": rem_load,
                "removed_at": str(r.get("at") or ""),
                "why": str(r.get("by") or ""),
            })
    unreplaced_removals.reverse()

    # Completed renders in data/*_job/*.render.built.json newer than a laid row on the same (shot, subject) with no negative verdict
    newer_unlaid_renders: list[dict] = []
    data_root = vdir.parent
    completed_renders: dict[str, dict] = {}
    if data_root.is_dir():
        for bp in sorted(data_root.glob("*_job/*.render.built.json")):
            try:
                bdoc = json.loads(bp.read_text())
            except (OSError, ValueError):
                continue
            ld = bdoc.get("load") or {}
            rname = str(ld.get("name") or bp.name.split(".")[0])
            qp = bp.with_name(f"{rname}.render.queued.json")
            if not qp.is_file():
                continue
            try:
                qdoc = json.loads(qp.read_text())
            except (OSError, ValueError):
                continue
            if qdoc.get("status") not in ("success", "completed") and not qdoc.get("files"):
                continue
            out_folder = str(bdoc.get("output_folder") or "")
            cap_str = str(bdoc.get("capture") or "")
            if video and f"/{video}/" not in out_folder and f"_{video}_" not in cap_str and not out_folder.startswith(f"Video/{video}/"):
                continue
            rfirst = ld.get("first")
            rframes = ld.get("frames")
            rsub = str(ld.get("subject") or "")
            if rfirst is None or not rframes or not rsub:
                continue
            qat = str(qdoc.get("queued_at") or "")
            completed_renders[rname] = {
                "load": rname,
                "subject": rsub,
                "first": int(rfirst),
                "last": int(rfirst) + int(rframes) - 1,
                "queued_at": qat,
            }
    removed_loads = {str((r.get("removed") or {}).get("load") or "") for r in active_rem_rows} - set(laid_by_load.keys())
    for x in laid_rows:
        x_load = str(x["load"])
        x_sub = str(x["subject"])
        x_spans = _spans_of_laid(x)
        x_fset = _frame_set(x_spans)
        x_rec = completed_renders.get(x_load)
        x_qat = _norm_iso(x_rec["queued_at"]) if x_rec else _laid_effective_at(x, choices)
        for rname, rinfo in sorted(completed_renders.items()):
            if rname == x_load or rname in laid_by_load or rname in removed_loads or rname in _extract_replaces(x):
                continue
            if rinfo["subject"] != x_sub:
                continue
            rfset = set(range(rinfo["first"], rinfo["last"] + 1))
            if not (x_fset & rfset):
                continue
            if not rinfo["queued_at"] or not x_qat or _norm_iso(rinfo["queued_at"]) <= x_qat:
                continue
            rch = choices.get(rname)
            if rch is not None and not _is_keep(rch["chosen"]):
                continue
            newer_unlaid_renders.append({
                "shot": f"{x['first']}-{x['last']}",
                "subject": x_sub,
                "laid_load": x_load,
                "newer_load": rname,
                "newer_span": [[rinfo["first"], rinfo["last"]]],
                "newer_queued_at": rinfo["queued_at"],
                "last_owner_choice": rch["chosen"] if rch else "none",
            })

    share_opt: Path | None = None
    sh_str = (getattr(a, "share", None) or os.environ.get("H3_OUTPUT_DIR") or "").strip()
    if sh_str:
        share_opt = Path(sh_str)
    elif (vdir / "inputs.json").is_file():
        try:
            icfg = json.loads((vdir / "inputs.json").read_text())
            if icfg.get("share") or icfg.get("output_dir"):
                share_opt = Path(str(icfg.get("share") or icfg.get("output_dir")))
        except Exception:
            pass

    prov = audit_cross_subject_provenance(
        laid_rows, laid_path, video, share=share_opt, data_arg=getattr(a, "data", None), active_rem_rows=active_rem_rows
    )

    print(f"=== Laid rows in {laid_path} ({len(laid_rows)} rows across {len(by_shot)} shots) ===")
    for r in laid_report:
        print(
            f"  {r['line']:3d}  {r['shot']:9}  {_fmt_spans(r['frames']):11}  {r['subject']:6}  "
            f"{r['load']:18}  {r['owner']:21}  {r['last_owner_choice']:46}  {r['status']}"
        )
    if prov["carved_frames"]:
        print(f"\n=== Carved frames on same subject ({len(prov['carved_frames'])}) ===")
        for cf in prov["carved_frames"]:
            rem_s = _fmt_spans(cf["remaining_frames"]) if cf["remaining_frames"] else "carved whole"
            cby_s = "; ".join(f"{c.get('load')} on {_fmt_spans(c.get('frames') or [])}" for c in (cf.get("carved_by") or []))
            print(f"  {cf['load']:18}  {cf['subject']:6}  remaining={rem_s:15}  carved_by: {cby_s}")
    if prov["original_restored_over_replacement"]:
        print(f"\n=== Original restored over older replacement ({len(prov['original_restored_over_replacement'])}) ===")
        for orr in prov["original_restored_over_replacement"]:
            orr_s = "; ".join(f"{k.get('older_load')} on {_fmt_spans(k.get('frames') or [])} ({k.get('why')})" for k in orr["original_restored_over_replacement"])
            print(f"  {orr['load']:18}  {orr['subject']:6}  frames={_fmt_spans(orr['frames']):15}  restored_over: {orr_s}")
    if prov["leaves_as_the_original_not_known"]:
        print(f"\n=== Rows where leaves_as_the_original was not_known ({len(prov['leaves_as_the_original_not_known'])}) ===")
        for nk in prov["leaves_as_the_original_not_known"]:
            print(f"  {nk['load']:18}  {nk['subject']:6}  frames={_fmt_spans(nk['frames'])}")
    if prov["stale_pieces"]:
        print(f"\n=== Stale pieces: made on an older piece of another subject ({len(prov['stale_pieces'])}) ===")
        for sp in prov["stale_pieces"]:
            print(
                f"  {sp['load']:18} ({sp['subject']:6}) on {_fmt_spans(sp['frames']):11}: source cut {sp['source_cut']} "
                f"held {sp['other_subject']}'s {sp['source_piece']}, now laid is {sp['current_laid_piece']}"
            )
    if prov["neither_made_on_the_other"]:
        print(f"\n=== Overlapping subjects where neither was made on a cut holding the other ({len(prov['neither_made_on_the_other'])}) ===")
        for nm in prov["neither_made_on_the_other"]:
            print(
                f"  {nm['loads'][0]} ({nm['subjects'][0]}) & {nm['loads'][1]} ({nm['subjects'][1]}) "
                f"on {_fmt_spans(nm['frames'])} (sources: {nm['sources'][0]}, {nm['sources'][1]})"
            )
    if prov["source_has_no_record"]:
        print(f"\n=== Multi-subject rows whose source cut has no sidecar record ({len(prov['source_has_no_record'])}) ===")
        for nr in prov["source_has_no_record"]:
            print(f"  {nr['load']:18} ({nr['subject']:6}) on {_fmt_spans(nr['frames']):11}: {nr['why']} (source={nr['source']})")
    if superseded_keeps:
        print(f"\n=== Historical keeps superseded ({len(superseded_keeps)}) ===")
        for s in superseded_keeps:
            print(
                f"  {s['load']:18}  {s['subject']:6}  {_fmt_spans(s['frames']):11}  "
                f"kept {s['keep_at'] or 'no-at':20} ({s['cite']:24}) -> superseded by {s['superseded_by']}"
            )
    if unreplaced_removals:
        print(f"\n=== Removed rows not yet replaced in laid.jsonl ({len(unreplaced_removals)}) ===")
        for u in unreplaced_removals:
            print(
                f"  {u['shot']:9}  {_fmt_spans(u['frames']):11}  {u['subject']:6}  "
                f"was {u['removed_load']:18} (removed {u['removed_at']}: {u['why']})"
            )
    if newer_unlaid_renders:
        print(f"\n=== Newer completed renders without negative verdict ({len(newer_unlaid_renders)}) ===")
        for n_ren in newer_unlaid_renders:
            print(
                f"  {n_ren['shot']:9}  {n_ren['subject']:6}  laid {n_ren['laid_load']:18} < newer {n_ren['newer_load']:18} "
                f"({_fmt_spans(n_ren['newer_span'])}, queued {n_ren['newer_queued_at']}, choice={n_ren['last_owner_choice']})"
            )
    summary = (
        f"{len(laid_rows)} laid rows across {len(by_shot)} shots "
        f"({len(multi_sub_shots)} multi-subject composite shots, {len(split_span_shots)} split-span subject shots); "
        f"{len(kept_loads)} kept loads verified; {len(superseded_keeps)} historical keeps superseded; "
        f"{len(unreplaced_removals)} unreplaced removals; {len(newer_unlaid_renders)} newer unlaid renders; "
        f"{len(missing_keeps)} kept loads missing from laid.jsonl; {len(wrong_laid)} laid loads marked wrong"
    )
    print(f"\n{summary}")
    if getattr(a, "json", None):
        Path(a.json).write_text(json.dumps({
            "laid_path": str(laid_path),
            "laid_rows": laid_report,
            "carved_frames": prov["carved_frames"],
            "original_restored_over_replacement": prov["original_restored_over_replacement"],
            "leaves_as_the_original_not_known": prov["leaves_as_the_original_not_known"],
            "stale_pieces": prov["stale_pieces"],
            "neither_made_on_the_other": prov["neither_made_on_the_other"],
            "source_has_no_record": prov["source_has_no_record"],
            "superseded_keeps": superseded_keeps,
            "unreplaced_removals": unreplaced_removals,
            "newer_unlaid_renders": newer_unlaid_renders,
            "missing_keeps": missing_keeps,
            "wrong_laid": wrong_laid,
            "counts": {
                "laid_rows": len(laid_rows),
                "shots": len(by_shot),
                "multi_subject_shots": len(multi_sub_shots),
                "split_span_subject_shots": len(split_span_shots),
                "kept_loads_verified": len(kept_loads),
                "historical_keeps_superseded": len(superseded_keeps),
                "unreplaced_removals": len(unreplaced_removals),
                "newer_unlaid_renders": len(newer_unlaid_renders),
                "kept_loads_missing": len(missing_keeps),
                "laid_loads_marked_wrong": len(wrong_laid),
                "carved_frames": len(prov["carved_frames"]),
                "original_restored_over_replacement": len(prov["original_restored_over_replacement"]),
                "stale_pieces": len(prov["stale_pieces"]),
                "neither_made_on_the_other": len(prov["neither_made_on_the_other"]),
                "source_has_no_record": len(prov["source_has_no_record"]),
            },
            "summary": summary,
        }, indent=2) + "\n")
    if wrong_laid or missing_keeps:
        msgs: list[str] = []
        for w in wrong_laid:
            msgs.append(
                f"LAID_MARKED_WRONG: line {w['line']} of {laid_path} lays {w['load']} on {w['shot']} "
                f"(frames {_fmt_spans(w['frames'])}, {w['subject']}, owner={w['owner']!r}), "
                f"but the owner's last choice on {w['load']} is {w['last_owner_choice']}"
            )
        for m_keep in missing_keeps:
            msgs.append(
                f"KEEP_NOT_LAID: {m_keep['load']} ({m_keep['subject']}, frames {_fmt_spans(m_keep['frames'])}) "
                f"has last owner choice keep at {m_keep['keep_at'] or 'no-at'} ({m_keep['cite']}), "
                f"is not in {laid_path}, and is not superseded by any newer laid row on its frames"
            )
        raise SystemExit("\n".join(msgs))
    return 0


def _add_write_args(sp: argparse.ArgumentParser, is_remove: bool = False, is_replace: bool = False) -> None:
    sp.add_argument("--video", help="video stem (defaults to the parent folder of --laid)")
    sp.add_argument("--laid", help="path to data/<video>/laid.jsonl (defaults to data/<video>/laid.jsonl)")
    sp.add_argument("--share", help="the server's output folder (defaults to H3_OUTPUT_DIR or inputs.json)")
    sp.add_argument("--shot", help="shot span as FIRST-LAST")
    sp.add_argument("--first", type=int, help="first frame of the shot")
    sp.add_argument("--last", type=int, help="last frame of the shot")
    sp.add_argument("--subject", required=not is_remove, help="subject name")
    sp.add_argument("--load", required=not is_remove, help="render load name")
    if is_remove:
        sp.add_argument("--by", required=True, help="why the line comes out; kept with the line whole in laid.removed.jsonl beside the list")
    if not is_remove:
        sp.add_argument("--file", required=True, help="render file relative to --share (or absolute inside --share)")
        sp.add_argument("--owner", choices=OWNER, required=True, help="owner state, always said (no default since 2026-10-21): "
                        "`kept` only where the owner's last choice on this render is a keep")
        sp.add_argument("--by", required=True, help="chooser's words / reason")
        sp.add_argument("--owner-word", metavar="FILE:LINE", help="for `kept` and `kept in a joined clip`: the owner's own line saying the "
                        "keep (data/<video>/owner_answers.jsonl or data/<video>/verdicts/verdicts.jsonl, by line number), read and held to it")
        sp.add_argument("--frames", help="frames laid, e.g. [[first, last]] or FIRST-LAST: the delivery table lays the render on these "
                        "frames only and says the rest of the shot is left as the source (since 2026-10-21)")
        sp.add_argument("--allow-partial-lay", nargs="?", const="", default=None, metavar="WHY",
                        help="allow laying --frames narrower than the shot when the remaining shot frames are not yet covered (records partial_lay_why)")
        sp.add_argument("--allow-parallel-raw", nargs="?", const="", default=None, metavar="WHY",
                        help="allow laying a piece whose job was rendered on a raw source or cut missing another subject's laid piece on the same frames (records parallel_raw_why)")
        sp.add_argument("--added", help="UTC timestamp (default: now)")
        sp.add_argument("--added-by", help="session name (default: H3_SESSION or --by)")
    if is_replace:
        sp.add_argument("--replaces", help="specific load on that shot and subject to replace (default: any line on that shot and subject)")
    sp.add_argument("--no-rebuild", nargs="?", const="", default=None, metavar="WHY",
                    help="skip running bench/rebuild_delivery.py for a batch of lays (requires WHY, recorded in the line; for remove, in laid.removed.jsonl)")
    sp.add_argument("--inputs", help="optional inputs.json path forwarded to bench/rebuild_delivery.py")
    sp.add_argument("--data", help="optional data/<video> path forwarded to bench/rebuild_delivery.py")
    sp.add_argument("--out-dir", help="optional output folder forwarded to bench/rebuild_delivery.py")
    sp.add_argument("--day", help="optional day folder forwarded to bench/rebuild_delivery.py")
    sp.add_argument("--stamp", help="optional stamp forwarded to bench/rebuild_delivery.py")
    sp.add_argument("--source", help="optional source video forwarded to bench/rebuild_delivery.py")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed", help="write the list that reproduces what a table laid")
    s.add_argument("--table", required=True, help="a delivery table's JSON (bench/delivery_table.py --json)")
    s.add_argument("--share", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--by", required=True, help="the session that seeds it")
    _add_write_args(sub.add_parser("lay", help="append a laid piece and rebuild the delivery table and pages"))
    _add_write_args(sub.add_parser("replace", help="replace a piece on a shot and subject and rebuild the delivery table and pages"), is_replace=True)
    _add_write_args(sub.add_parser("remove", help="remove a laid line and rebuild the delivery table and pages"), is_remove=True)
    so = sub.add_parser("set-owner", help="change one line's owner state in place, keeping what it said before in its history")
    so.add_argument("--video", help="video stem (defaults to the parent folder of --laid)")
    so.add_argument("--laid", help="path to data/<video>/laid.jsonl (defaults to data/<video>/laid.jsonl)")
    so.add_argument("--share", help="the server's output folder (defaults to H3_OUTPUT_DIR or inputs.json)")
    so.add_argument("--load", required=True, help="the line's load")
    so.add_argument("--shot", help="the line's shot as FIRST-LAST, where the load is laid on more than one")
    so.add_argument("--subject", help="the line's subject, where the load is laid for more than one on the shot")
    so.add_argument("--first", type=int, help=argparse.SUPPRESS)
    so.add_argument("--last", type=int, help=argparse.SUPPRESS)
    so.add_argument("--owner", choices=OWNER, required=True, help="the new owner state")
    so.add_argument("--by", required=True, help="where the owner's word is, or why the line is not the owner's word")
    so.add_argument("--owner-word", metavar="FILE:LINE", help="for `kept` and `kept in a joined clip`: the owner's own line saying the keep "
                    "(data/<video>/owner_answers.jsonl or data/<video>/verdicts/verdicts.jsonl, by line number)")
    so.add_argument("--session", help="who makes the change (default: H3_SESSION)")
    so.add_argument("--no-rebuild", nargs="?", const="", default=None, metavar="WHY",
                    help="skip running bench/rebuild_delivery.py for a batch of changes (requires WHY, recorded in the line's history entry)")
    for flag in ("--inputs", "--data", "--out-dir", "--day"):
        so.add_argument(flag, help="optional, forwarded to bench/rebuild_delivery.py")
    sy = sub.add_parser("sync", help="sync laid.jsonl with all standing owner keep clicks in chronological order")
    sy.add_argument("--video", help="video stem (defaults to the parent folder of --laid)")
    sy.add_argument("--laid", help="path to data/<video>/laid.jsonl (defaults to data/<video>/laid.jsonl)")
    sy.add_argument("--share", help="the server's output folder (defaults to H3_OUTPUT_DIR or inputs.json)")
    sy.add_argument("--answers", help="path to data/<video>/owner_answers.jsonl")
    sy.add_argument("--verdicts", help="path to data/<video>/verdicts/verdicts.jsonl")
    sy.add_argument("--data", help="optional data/<video> path")
    sy.add_argument("--json", help="optional path to write JSON sync report")
    v = sub.add_parser("verify", help="print every line with its file's state now")
    v.add_argument("--laid", required=True)
    v.add_argument("--share", required=True)
    kp = sub.add_parser("keeps", help="verify every laid row and every owner keep against owner_answers.jsonl and verdicts.jsonl")
    kp.add_argument("--video", help="video stem (defaults to the parent folder of --laid)")
    kp.add_argument("--laid", help="path to data/<video>/laid.jsonl (defaults to data/<video>/laid.jsonl)")
    kp.add_argument("--answers", help="path to data/<video>/owner_answers.jsonl")
    kp.add_argument("--verdicts", help="path to data/<video>/verdicts/verdicts.jsonl")
    kp.add_argument("--removed", help="path to data/<video>/laid.removed.jsonl")
    kp.add_argument("--data", help="optional data/<video> path")
    kp.add_argument("--share", help="optional server output folder for cut sidecar lookup")
    kp.add_argument("--json", help="optional path to write JSON verification report")
    rf = sub.add_parser("refs", help="every line's reference as its render read it, against the file now")
    rf.add_argument("--laid", required=True)
    rf.add_argument("--ledger", required=True)
    rf.add_argument("--share", required=True)
    rf.add_argument("--json", help="the rows as JSON here")
    a = p.parse_args(argv)
    if a.cmd in ("lay", "replace", "remove"):
        return mutate_laid(a)
    if a.cmd == "set-owner":
        return set_owner(a)
    if a.cmd == "sync":
        return sync(a)
    if a.cmd == "keeps":
        return keeps(a)
    if a.cmd == "refs":
        rows = refs(read(a.laid, a.share, check_bytes=False), json.loads(Path(a.ledger).read_text()), HERE.parent)
        for x in rows:
            print(f"{x['line']:3d} {x['first']:>4}-{x['last']:<4} {x['subject']:7} {x['load']:22} {x['state']:42} {x['reference'] or ''}")
        if a.json:
            Path(a.json).write_text(json.dumps(rows, indent=1) + "\n")
        flagged = [x for x in rows if not x["state"].startswith(("same bytes", "no digest kept; older"))]
        print(f"{len(rows)} reference rows for {len(set(x['line'] for x in rows))} lines; {len(flagged)} not shown to be the bytes the render read")
        return 0
    if a.cmd == "seed":
        if Path(a.out).exists():
            raise SystemExit(f"{a.out} exists: the list is changed by a new line written by hand, never seeded over")
        lines = seed(a.table, a.share, a.by)
        Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in lines))
        print(f"wrote {a.out}: {len(lines)} lines from {a.table}; " + "; ".join(f"{sum(x['owner'] == o for x in lines)} {o}" for o in OWNER))
        return 0
    lines = read(a.laid, a.share, check_bytes=False)
    bad = 0
    for x in lines:
        f = Path(a.share) / x["file"]
        state = "MISSING" if not f.is_file() else ("bytes as laid" if sha256(f) == x["sha256"] else "CHANGED since laid")
        if state == "bytes as laid" and x.get("zoomed_sha256") and sha256(zoomed(f)) != x["zoomed_sha256"]:
            state = "ZOOMED COMPANION CHANGED"
        bad += state != "bytes as laid"
        print(f"{x['line']:3d} {x['first']:>4}-{x['last']:<4} {x['subject']:7} {x['load']:22} {x['owner']:21} {state}")
    counts = "; ".join(f"{sum(x['owner'] == o for x in lines)} {o}" for o in OWNER)
    print(f"{len(lines)} lines in {a.laid} (sha256 {digest(a.laid)[:16]}): {counts}; {bad} missing or changed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
