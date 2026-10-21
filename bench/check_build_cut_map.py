#!/usr/bin/env python3
"""`bench/build_cut_map.py` names the cut and gaps of the laid list as it is, says where they came from, and writes
nothing when asked for help.

  help_writes_nothing             `--help` prints the docstring and exits 0, and no file under the data root is written
                                  or touched (until 2026-10-21 it had no argparse, and `--help` rebuilt the owner's pages)
  table_chosen_by_laid_sha        of two tables with usable gap checks, the one whose `made` line names laid.jsonl's sha
                                  is read, though the other has the newer stamp and the newer mtime (it used to be the
                                  newest by mtime); of two tables of today's sha, the newer stamp is read though the
                                  older has the newer mtime
  no_match_is_said                with no table made from today's sha, nothing is called a match; the newest usable table
                                  is read as the fallback and the page's words say so, naming it and its laid sha
  gap_check_bound_to_its_table    a table of today's sha whose gap check names another table's sha256 is not read, and
                                  the page's words give `review_pages.read_gap_check`'s reason
  a_bad_record_is_a_reason        a half-written gap check, a table that is a JSON list, and a recorded sha too short to
                                  be one neither crash the build nor match: each is skipped, and a rejected table's
                                  reason is said
  cut_from_the_tables_own_out     the cut is the table's own `assembly.command --out`, its day folder kept (it used to be
                                  a name built from the stamp, with no folder)
  frames_against_the_list_now     frame by frame against the laid list in file order: a line the check did not measure
                                  is laid, not measured, on its `frames` only; a piece the check measured that is no
                                  longer laid, or no longer on those frames, is not carried there; a measured piece still
                                  laid keeps the check's figures, beside a new one and whatever range the new one's
                                  capture covers; of two lines on one frame the first in the file is shown
  read_from_names_sha_and_clock   every record read is listed with the sha256 of that file (16 hex; a video by its size)
                                  and a file time on the owner's clock
  page_names_its_tool_and_list    a whole run writes the three files and changes no other file under the data root, the
                                  page stamped `data-laid-sha256` with the list read and naming the file that ran it
  fallback_rows_say_laid          on the fallback, a laid row the check did not measure says LAID, not IN CUT, never
                                  "100% Carried", and is not counted clean

    <comfy venv python> bench/check_build_cut_map.py

No ffmpeg, no model, no card, no server, no share: a temporary data root with a few small records.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

TOOL = Path(__file__).resolve().parent / "build_cut_map.py"
import build_cut_map as bcm  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@contextmanager
def rooted(data: Path):
    """The module's folders pointed at a temporary data root for one case, and put back after (no case reads the
    repo's own data/)."""
    was = bcm.DATA_DIR, bcm.FUN_DIR
    bcm.DATA_DIR, bcm.FUN_DIR = data, data / "fun"
    try:
        yield
    finally:
        bcm.DATA_DIR, bcm.FUN_DIR = was


def laid_line(load: str, first: int, last: int, subject: str = "lead", frames=None) -> dict:
    line = {"schema": "h3_laid/1", "first": first, "last": last, "subject": subject, "load": load,
            "file": f"Video/v/{load}_00001.mp4", "owner": "unconfirmed", "by": "check"}
    if frames:
        line["frames"] = frames
    return line


def make_root(tmp: Path, laid: list[dict]) -> Path:
    data = tmp / "data"
    (data / "fun" / "cast").mkdir(parents=True)
    (data / "fun" / "poc").mkdir()
    (data / "fun" / "shots.json").write_text(json.dumps({"shots": [
        {"first": 0, "last": 9, "who": {"lead": {"present": "yes"}}}]}))
    (data / "fun" / "plan.json").write_text(json.dumps({"subjects": {"lead": {"parts": ["face_and_neck"]}}}))
    (data / "fun" / "cast" / "cast.jsonl").write_text("")
    (data / "fun" / "laid.jsonl").write_text("".join(json.dumps(x) + "\n" for x in laid))
    return data


def measured(pieces: list) -> dict:
    """A gap check's shot 0-9 for the lead: `pieces[i]` laid and carried on frame i, None where nothing is laid."""
    return {"shot": [0, 9], "subject": "lead", "frames": list(range(10)), "piece": list(pieces),
            "step_name": ["kept_and_laid" if x else "captured_not_rendered" for x in pieces],
            "state": ["carried" if x else "original: no piece laid" for x in pieces]}


def make_table(data: Path, share: Path, stamp: str, laid_sha16: str, *, cov_table_sha: str | None = None,
               shots: list[dict] | None = None, day: str | None = None) -> Path:
    """A whole-video table of `stamp` made from the list `laid_sha16`, with a gap check that passes
    `review_pages.read_gap_check` (or names `cov_table_sha` instead of the table's own sha256), and its cut."""
    day = day or stamp[:10]
    tdir = data / f"{day}_fun_delivery_tables"
    tdir.mkdir(exist_ok=True)
    cut = share / "Video" / "fun" / f"day_{day}" / f"fun_0000_6394_joined_{stamp}_from_laid.mp4"
    cut.parent.mkdir(parents=True, exist_ok=True)
    cut.write_bytes(b"not a video")
    check = Path(str(cut) + ".check.json")
    check.write_text(json.dumps({"verdict": "passes"}))
    cov = tdir / f"frame_coverage_fun_0000_6394_{stamp}.json"
    table = tdir / f"t_fun_0000_6394_{stamp}.json"
    table.write_text(json.dumps({
        "made": f"Made from ONLY the list of what is laid, `data/fun/laid.jsonl` (sha256 {laid_sha16}, 1 lines), and nothing else.",
        "assembly": {"command": f"assemble_delivery.py --table {table} --out {cut}", "stack": {"coverage_json": str(cov)}},
        "rows": [], "reading": []}))
    cov.write_text(json.dumps({
        "table": str(table), "table_sha256": cov_table_sha or sha(table), "span": None,
        "by_shot": {"remake": [], "capture": []}, "clip_check": {"file": str(check), "sha256": sha(check)},
        "shots": shots or []}))
    return table


def snapshot(root: Path) -> dict:
    return {str(p): (p.stat().st_mtime_ns, p.stat().st_size, sha(p)) for p in root.rglob("*") if p.is_file()}


def run_tool(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True,
                          env={**os.environ, "H3_OUTPUT_DIR": os.environ.get("H3_OUTPUT_DIR", "/nonexistent")})


def help_writes_nothing():
    with tempfile.TemporaryDirectory() as t:
        data = make_root(Path(t), [laid_line("a", 0, 9)])
        before = snapshot(Path(t))
        r = run_tool("--help", "--data", str(data), "--share", str(Path(t) / "share"))
        assert r.returncode == 0, f"--help exit {r.returncode}: {r.stderr[-400:]}"
        assert "Writes, and nothing else" in r.stdout, f"--help printed no docstring: {r.stdout[:200]!r}"
        after = snapshot(Path(t))
        assert after == before, f"--help wrote or touched: {sorted(set(after) ^ set(before)) or 'changed files'}"


def table_chosen_by_laid_sha():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = make_root(tmp, [laid_line("a", 0, 9)])
        today = sha(data / "fun" / "laid.jsonl")
        right = make_table(data, tmp / "share", "2026-10-20T1000Z", today[:16])
        wrong = make_table(data, tmp / "share", "2026-10-21T1000Z", "0" * 16)
        os.utime(wrong, (right.stat().st_mtime + 100, right.stat().st_mtime + 100))
        b = bcm.bound_records(data, data / "fun" / "laid.jsonl")
        assert b["match"] and b["match"]["table"] == right, f"read {b['match'] and b['match']['table'].name}, not {right.name}"
        assert b["laid_sha"] == today
        # two of today's sha: the newer stamp, though the other has the newer mtime
        newer = make_table(data, tmp / "share", "2026-10-21T1100Z", today[:16])
        os.utime(right, (newer.stat().st_mtime + 100, newer.stat().st_mtime + 100))
        b = bcm.bound_records(data, data / "fun" / "laid.jsonl")
        assert b["match"]["table"] == newer, f"of two of today's sha read {b['match']['table'].name}, not the newer stamp"


def no_match_is_said():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = make_root(tmp, [laid_line("a", 0, 9)])
        older = make_table(data, tmp / "share", "2026-10-19T1000Z", "1" * 16)
        newer = make_table(data, tmp / "share", "2026-10-20T1000Z", "2" * 16)
        b = bcm.bound_records(data, data / "fun" / "laid.jsonl")
        assert b["match"] is None, f"called {b['match'] and b['match']['table'].name} a match"
        assert b["fallback"]["table"] == newer, f"fallback {b['fallback']['table'].name}, not the newest usable {newer.name}"
        w = bcm.bound_words(b)[0]
        assert w.startswith("No cut on disk was made from laid.jsonl as it is now"), w
        assert newer.name in w and "2" * 16 in w, f"the words do not name the table read and its laid sha: {w}"
        assert older.name not in w


def gap_check_bound_to_its_table():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = make_root(tmp, [laid_line("a", 0, 9)])
        today = sha(data / "fun" / "laid.jsonl")
        bad = make_table(data, tmp / "share", "2026-10-21T1000Z", today[:16], cov_table_sha="f" * 64)
        b = bcm.bound_records(data, data / "fun" / "laid.jsonl")
        assert b["match"] is None, "read a gap check that names another table"
        assert [r["table"] for r in b["rejected"]] == [bad], f"rejected {b['rejected']}"
        words = " ".join(bcm.bound_words(b))
        assert "table_sha256" in words and bad.name in words, f"the reason is not said: {words}"


def a_bad_record_is_a_reason():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = make_root(tmp, [laid_line("a", 0, 9)])
        today = sha(data / "fun" / "laid.jsonl")
        half = make_table(data, tmp / "share", "2026-10-21T1000Z", today[:16])
        cov = data / "2026-10-21_fun_delivery_tables" / "frame_coverage_fun_0000_6394_2026-10-21T1000Z.json"
        cov.write_text('{"table_sha256": "')                              # half written
        (data / "2026-10-21_fun_delivery_tables" / "t_fun_0000_6394_2026-10-21T1100Z.json").write_text("[1, 2]")
        short = make_table(data, tmp / "share", "2026-10-21T1200Z", today[:1])   # a one-digit "sha"
        b = bcm.bound_records(data, data / "fun" / "laid.jsonl")
        assert b["match"] is None, f"matched {b['match'] and b['match']['table'].name}"
        assert [r["table"] for r in b["rejected"]] == [half], f"rejected {[r['table'].name for r in b['rejected']]}"
        assert short not in [r["table"] for r in b["rejected"]] and (b["fallback"] or {}).get("table") != short
        words = " ".join(bcm.bound_words(b))
        assert half.name in words and "Error" in words, f"the half-written check's reason is not said: {words}"


def cut_from_the_tables_own_out():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = make_root(tmp, [laid_line("a", 0, 9)])
        make_table(data, tmp / "share", "2026-10-21T1000Z", sha(data / "fun" / "laid.jsonl")[:16])
        b = bcm.bound_records(data, data / "fun" / "laid.jsonl")
        cut = b["match"]["cut"]
        assert cut == tmp / "share" / "Video" / "fun" / "day_2026-10-21" / "fun_0000_6394_joined_2026-10-21T1000Z_from_laid.mp4", cut
        assert cut.is_file()


def frames_against_the_list_now():
    with tempfile.TemporaryDirectory() as t:
        data = make_root(Path(t), [])
        cap = data / "2026-10-21_b"                                      # a capture of `b` that starts on frame 5
        cap.mkdir()
        (cap / "manifest.json").write_text(json.dumps({"first_frame": 5, "frames": 5}))
        with rooted(data):
            def cats(lrows, pieces):
                cov = bcm.build_frame_coverage_for_row(0, 9, "lead", lrows, {(0, 9, "lead"): measured(pieces)}, {})
                return [pf["cat"] for pf in cov["frames"]], cov
            # a line the check did not measure, laying 2-5 of its span, where the check measured `a`
            c, cov = cats([laid_line("b", 0, 9, frames=[[2, 5]])], ["a"] * 10)
            assert c[2:6] == ["not_measured"] * 4, f"2-5 not said as not measured: {c}"
            assert "carried" not in c, f"unmeasured or no-longer-laid frames said carried: {c}"
            assert cov["counts"]["not_measured"] == 4 and "not measured" in cov["short_summary"], cov["short_summary"]
            # nothing laid now, though the check measured `a`
            c, _ = cats([], ["a"] * 10)
            assert "carried" not in c, f"a piece taken out still reads carried: {c}"
            # `a` still laid but only on 0-4 now
            c, _ = cats([laid_line("a", 0, 9, frames=[[0, 4]])], ["a"] * 10)
            assert c == ["carried"] * 5 + ["unrendered_gap"] * 5, f"frames a line no longer lays read as the check had them: {c}"
            # still laid and measured: the check's figures stand
            c, _ = cats([laid_line("a", 0, 9)], ["a"] * 10)
            assert c == ["carried"] * 10, f"a measured, still-laid piece lost its figures: {c}"
            # a measured piece beside a new one whose capture starts on 5: the measured frames keep the check's figures
            c, _ = cats([laid_line("a", 0, 9, frames=[[0, 4]]), laid_line("b", 0, 9, frames=[[5, 9]])], ["a"] * 5 + [None] * 5)
            assert c == ["carried"] * 5 + ["not_measured"] * 5, f"measured and new pieces mixed up: {c}"
            # a check made from this very list stands frame by frame, "no track" included; from another list it does not
            untracked = measured(["a"] * 8 + [None] * 2)
            untracked["step_name"][8:] = ["no_capture"] * 2
            cov = bcm.build_frame_coverage_for_row(0, 9, "lead", [laid_line("a", 0, 9)], {(0, 9, "lead"): untracked}, {},
                                                   check_is_of_this_list=True)
            got = [pf["cat"] for pf in cov["frames"]]
            assert got == ["carried"] * 8 + ["no_capture"] * 2, f"a check of this list was second-guessed: {got}"
            cov = bcm.build_frame_coverage_for_row(0, 9, "lead", [laid_line("a", 0, 9)], {(0, 9, "lead"): untracked}, {})
            assert [pf["cat"] for pf in cov["frames"]][8:] == ["not_measured"] * 2, "a check of another list taken as is"
            # two unmeasured lines on the same frames: the first in the file lays them
            _, cov = cats([laid_line("x", 0, 9), laid_line("y", 0, 9)], [None] * 10)
            assert {pf["piece"] for pf in cov["frames"]} == {"x"}, f"not the first line in the file: {[pf['piece'] for pf in cov['frames']]}"


def read_from_names_sha_and_clock():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = make_root(tmp, [laid_line("a", 0, 9)])
        table = make_table(data, tmp / "share", "2026-10-21T1000Z", sha(data / "fun" / "laid.jsonl")[:16])
        with rooted(data):
            b = bcm.bound_records(data, data / "fun" / "laid.jsonl")
            rows = {r["what"]: r for r in bcm.read_from_rows(b, b["match"])}
        want = {"the list of what is laid": data / "fun" / "laid.jsonl", "the delivery table": table,
                "the gap check": b["match"]["cov_path"], "the shots record": data / "fun" / "shots.json"}
        for what, p in want.items():
            assert what in rows, f"{what} not listed: {sorted(rows)}"
            assert rows[what]["sha"] == sha(p)[:16], f"{what}: listed {rows[what]['sha']}, the file's is {sha(p)[:16]}"
        assert rows["the cut"]["sha"].endswith("not hashed"), rows["the cut"]
    # a recorded `data/...` path is read under a --data root of any name, and a malformed shot entry is skipped
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "elsewhere"
        (root / "x").mkdir(parents=True)
        with rooted(root):
            assert bcm.from_record("data/x") == root / "x", f"read {bcm.from_record('data/x')}, not under the --data root"
        assert bcm.coverage_index({"shots": [None, {"shot": [0, 9], "subject": "lead"}, {"shot": "x"}]}) == {
            (0, 9, "lead"): {"shot": [0, 9], "subject": "lead"}}
        for r in rows.values():
            assert r["mtime"] == "—" or re.search(r" C[DS]T$", r["mtime"]), f"not on the owner's clock: {r}"


def page_names_its_tool_and_list():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = make_root(tmp, [laid_line("a", 0, 9)])
        today = sha(data / "fun" / "laid.jsonl")
        make_table(data, tmp / "share", "2026-10-21T1000Z", today[:16])
        before = snapshot(tmp)
        r = run_tool("--data", str(data), "--share", str(tmp / "share"))
        assert r.returncode == 0, f"exit {r.returncode}: {(r.stdout + r.stderr)[-600:]}"
        after = snapshot(tmp)
        outs = {str(data / "fun" / p) for p in ("CUT_MAP.md", "poc/CUT_MAP.html", "poc/PROVENANCE.html")}
        assert set(after) - set(before) == outs, f"wrote {sorted(set(after) - set(before))}"
        changed = [p for p in before if after.get(p) != before[p]]
        assert not changed, f"changed files it only reads: {changed}"
        page = (data / "fun" / "poc" / "PROVENANCE.html").read_text()
        assert f'data-laid-sha256="{today}"' in page, "the page carries no data-laid-sha256 of the list it read"
        assert "bench/build_cut_map.py (sha256 " in page, "the page does not name the file that ran it"
        assert "were made from laid.jsonl as it is now" in page


def fallback_rows_say_laid():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = make_root(tmp, [laid_line("b", 0, 9)])                   # laid now: `b`, which no check measured
        make_table(data, tmp / "share", "2026-10-20T1000Z", "3" * 16, shots=[measured(["a"] * 10)])
        r = run_tool("--data", str(data), "--share", str(tmp / "share"))
        assert r.returncode == 0, f"exit {r.returncode}: {(r.stdout + r.stderr)[-600:]}"
        md = (data / "fun" / "CUT_MAP.md").read_text()
        row = next((ln for ln in md.splitlines() if ln.startswith("| `0–9`") and "`lead`" in ln), "")
        assert row, "no row for 0-9 lead"
        assert "LAID (" in row and "IN CUT" not in row, f"the row does not say LAID: {row[:200]}"
        assert "100% Carried" not in row and "not measured" in row, f"an unmeasured row reads carried: {row[:200]}"
        assert "**0** clean" in md, "an unmeasured row is counted clean"
        assert "**1 / 1** laid (" in md and "in cut" not in md.split("Subject-Shot Coverage", 1)[1].split("\n", 1)[0], \
            "the fallback's summary still says in cut"
        assert "100% Carried" not in md, "a label claims 100% where the figure is not read off the frames"
    # a laid row with frames the check found no track on, on a match: a gap, not clean
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        data = make_root(tmp, [laid_line("a", 0, 9)])
        untracked = measured(["a"] * 8 + [None] * 2)
        untracked["step_name"][8:] = ["no_capture"] * 2
        make_table(data, tmp / "share", "2026-10-20T1000Z", sha(data / "fun" / "laid.jsonl")[:16], shots=[untracked])
        r = run_tool("--data", str(data), "--share", str(tmp / "share"))
        assert r.returncode == 0, f"exit {r.returncode}: {(r.stdout + r.stderr)[-600:]}"
        md = (data / "fun" / "CUT_MAP.md").read_text()
        row = next((ln for ln in md.splitlines() if ln.startswith("| `0–9`") and "`lead`" in ln), "")
        assert "**0** clean" in md and "8/10f carried" in row and "no track" in row, f"untracked frames counted clean: {row[:200]}"
    # a clean row, measured carried, on the fallback and on a match: its label gives the figure, and its badge the word
    for matched in (False, True):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            data = make_root(tmp, [laid_line("a", 0, 9)])
            today = sha(data / "fun" / "laid.jsonl")
            make_table(data, tmp / "share", "2026-10-20T1000Z", today[:16] if matched else "3" * 16, shots=[measured(["a"] * 10)])
            r = run_tool("--data", str(data), "--share", str(tmp / "share"))
            assert r.returncode == 0, f"exit {r.returncode}: {(r.stdout + r.stderr)[-600:]}"
            md = (data / "fun" / "CUT_MAP.md").read_text()
            row = next((ln for ln in md.splitlines() if ln.startswith("| `0–9`") and "`lead`" in ln), "")
            word = "IN CUT" if matched else "LAID"
            assert f"{word} (unconfirmed · 10/10f carried)" in row, f"{'match' if matched else 'fallback'}: {row[:200]}"
            page = (data / "fun" / "poc" / "PROVENANCE.html").read_text()
            assert f"'>{word}</span>" in page and (matched or "'>IN CUT" not in page), \
                f"{'match' if matched else 'fallback'}: the row's badge does not say {word}"


for name, fn in [("help_writes_nothing", help_writes_nothing), ("table_chosen_by_laid_sha", table_chosen_by_laid_sha),
                 ("no_match_is_said", no_match_is_said), ("gap_check_bound_to_its_table", gap_check_bound_to_its_table),
                 ("a_bad_record_is_a_reason", a_bad_record_is_a_reason),
                 ("cut_from_the_tables_own_out", cut_from_the_tables_own_out),
                 ("frames_against_the_list_now", frames_against_the_list_now),
                 ("read_from_names_sha_and_clock", read_from_names_sha_and_clock),
                 ("page_names_its_tool_and_list", page_names_its_tool_and_list),
                 ("fallback_rows_say_laid", fallback_rows_say_laid)]:
    case(name, fn)
sys.exit(finish())
