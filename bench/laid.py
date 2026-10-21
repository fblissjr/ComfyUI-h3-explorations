#!/usr/bin/env python3
"""The list of what is laid: one JSON line a piece, the ONLY thing the delivery table reads for what stands.

    <python> bench/laid.py seed --table T.json --share SHARE --out data/<video>/laid.jsonl --by NAME
    <python> bench/laid.py lay --video <video> --first F --last L --subject SUB --load LOAD --file FILE --owner OWNER [--owner-word FILE:LINE] --by WHY [--no-rebuild WHY]
    <python> bench/laid.py replace --video <video> --first F --last L --subject SUB --load LOAD --file FILE --owner OWNER [--owner-word FILE:LINE] --by WHY [--replaces OLD] [--no-rebuild WHY]
    <python> bench/laid.py set-owner --video <video> --load LOAD [--shot F-L] [--subject SUB] --owner OWNER [--owner-word FILE:LINE] --by WHY [--no-rebuild WHY]
    <python> bench/laid.py remove --video <video> [--first F --last L] [--subject SUB] [--load LOAD] --by WHY [--no-rebuild WHY]
    <python> bench/laid.py verify --laid data/<video>/laid.jsonl --share SHARE
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
`no_rebuild`. On one shot and subject the lines are laid in file order, each on frames no line before it took.

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
        "frames": _parse_frames(a.frames, first, last),
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


def mutate_laid(a: argparse.Namespace) -> int:
    why_no_rebuild: str | None = None
    if a.no_rebuild is not None:
        why_no_rebuild = a.no_rebuild.strip()
        if not why_no_rebuild or why_no_rebuild.startswith("-"):
            raise SystemExit("--no-rebuild requires a non-empty reason (WHY), recorded in the line, or for remove in laid.removed.jsonl")
    laid_path, video = _resolve_laid_and_video(a)
    share = _resolve_share(a, laid_path)
    existing = [{k: v for k, v in x.items() if k != "line"} for x in read(str(laid_path), str(share), check_bytes=True)] if laid_path.is_file() else []
    if a.cmd in ("replace", "remove") and not laid_path.is_file():
        raise SystemExit(f"{a.cmd}: {laid_path} is missing")

    if a.cmd == "lay":
        first, last = _parse_shot(a, required=True)
        assert first is not None and last is not None
        entry = _build_entry(a, share, first, last, why_no_rebuild)
        new_lines = [*existing, entry]
        action_desc = f"laid {entry['load']} on {first}-{last} ({entry['subject']})"
    elif a.cmd == "replace":
        first, last = _parse_shot(a, required=True)
        assert first is not None and last is not None
        target_load = (getattr(a, "replaces", None) or "").strip()
        kept_lines, replaced_loads = [], []
        for x in existing:
            same_slot = int(x["first"]) == first and int(x["last"]) == last and x["subject"] == a.subject
            if same_slot and (not target_load or x["load"] == target_load):
                replaced_loads.append(str(x["load"]))
            else:
                kept_lines.append(x)
        if not replaced_loads:
            what = f"load {target_load!r} on {first}-{last} ({a.subject})" if target_load else f"any line on {first}-{last} ({a.subject})"
            raise SystemExit(f"replace: {laid_path} has no line for {what} to replace")
        entry = _build_entry(a, share, first, last, why_no_rebuild, replaces=replaced_loads)
        new_lines = [*kept_lines, entry]
        action_desc = f"replaced {', '.join(replaced_loads)} with {entry['load']} on {first}-{last} ({entry['subject']})"
    elif a.cmd == "remove":
        first, last = _parse_shot(a, required=False)
        sub = (getattr(a, "subject", None) or "").strip()
        load = (getattr(a, "load", None) or "").strip()
        if first is None and not sub and not load:
            raise SystemExit("remove: give --load and/or --first --last (or --shot) and --subject")
        kept_lines, removed = [], []
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
        action_desc = "removed " + ", ".join(f"{x['load']} on {x['first']}-{x['last']} ({x['subject']})" for x in removed)
    else:
        raise SystemExit(f"unknown command {a.cmd!r}")

    laid_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = laid_path.with_suffix(".jsonl.tmp")
    tmp_path.write_text("".join(json.dumps(x) + "\n" for x in new_lines))
    read(str(tmp_path), str(share), check_bytes=True)
    sha_before = digest(str(laid_path)) if laid_path.is_file() else None
    if a.cmd == "remove":
        # A REMOVED LINE LEAVES ITS RECORD (2026-10-21, W3e): the line whole, why it went and who took it out, appended to
        # `laid.removed.jsonl` beside the list. Until then the reason was nowhere, and `--no-rebuild`'s was written onto
        # whichever line was last in the list
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        session = (os.environ.get("H3_SESSION") or "").strip() or "not said (no H3_SESSION)"
        with open(removed_path(laid_path), "a") as f:
            for x in removed:
                # written before the list is replaced, so a failure between the two can leave a record of a removal that did not
                # happen, never a line gone with no record
                f.write(json.dumps({"schema": REMOVED_SCHEMA, "removed": x, "by": a.by.strip(), "at": now, "by_session": session,
                                    "laid_sha256_before": sha_before, "laid_sha256_after": digest(str(tmp_path)),
                                    **({"no_rebuild": why_no_rebuild} if why_no_rebuild else {})}) + "\n")
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


def owner_word(cite: str, load: str, owner: str, cmd: str = "set-owner") -> str:
    """`FILE:LINE`, read and held to being the owner's last choice on its row (by `at`) and a keep (of this load's
    row, for `kept`; of a joined clip's row, a `joined` part in its name, for `kept in a joined clip`). Returns the
    cite as given. Lines are counted as `grep -n` counts them, blank ones included.

    What it cannot see: a later answer the owner gave in another file (a keep cited from the verdicts record is not
    held to answers not yet copied into it), and whether the joined clip it names laid this load."""
    path, sep, num = cite.rpartition(":")
    if not sep or not path or not num.isdigit() or int(num) < 1:
        raise SystemExit(f"{cmd}: --owner-word {cite!r} is not FILE:LINE")
    if not Path(path).is_file():
        raise SystemExit(f"{cmd}: --owner-word {cite!r}: {path} is not a file")
    lines = Path(path).read_text().split("\n")
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
    v = sub.add_parser("verify", help="print every line with its file's state now")
    v.add_argument("--laid", required=True)
    v.add_argument("--share", required=True)
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
