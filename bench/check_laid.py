#!/usr/bin/env python3
"""`bench/laid.py`'s owner state on a made list: who says a piece is kept, and what a change of that leaves on the line.

  owner_is_always_said          `lay` and `replace` refuse a line with no `--owner`, and `kept` with no `--owner-word`, and
                                write nothing (until 2026-10-21 `--owner` defaulted to `kept`, and five pieces laid on a go
                                to render read `kept`); with the owner's keep cited, `lay` writes `kept` and the citation
  set_owner_changes_only_owner  `set-owner` rewrites one line in place: `owner`, `by`, `owner_word`, `history` and nothing
                                else; its place, `replaces`, `added`, `frames`, `no_rebuild`, the sha256 and every other
                                line stand
  kept_needs_the_owners_keep    `kept` refused without `--owner-word`; on a line that is not the owner's, not a keep (words,
                                a "wrong", a prompt choice that begins "keep"), a keep of another row, a keep the owner later
                                took back on the same row, a blank line, a line that is not there; `kept in a joined clip` on
                                a row that is not a joined clip's; a keep whose row has a later "wrong" by `at` though
                                earlier in the file; `unconfirmed` with a citation; the state the line has; a
                                load on two subjects of one shot without `--subject`. Each refusal leaves the file's bytes
  a_removed_line_leaves_its_record  `remove` refuses no `--by`; keeps the line whole with why in `laid.removed.jsonl`; with
                                `--no-rebuild` no other line of the list changes
  history_grows                 kept and back to unconfirmed: two history entries, the citation gone from the line and kept
                                in the history; a keep in the verdicts record's form (`what: words`, no `who`) taken past an
                                older "not sure" appended after it; `kept in a joined clip` on a row ending `_joined` taken
  keeps_verifies_and_refuses    `keeps` passes on disjoint frame spans for the same subject on one shot (`1466-1534` and
                                `1575-1729`) and on historical keeps superseded by newer keeps or `replaces`; refuses
                                (`KEEP_NOT_LAID`) when a kept render is not laid and not covered on its frames, and refuses
                                (`LAID_MARKED_WRONG`) when a laid row's load was later marked wrong by the owner

    <comfy venv python> bench/check_laid.py

No ffmpeg, no model, no card, no server: a temporary share with small files standing in for renders, and every write
with `--no-rebuild`.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish, refused  # noqa: E402

LAID = Path(__file__).resolve().parent / "laid.py"
#: The owner's answers, as a page writes them; line numbers below are these lines' (line 7 is blank).
ANSWERS = [
    {"who": "owner", "row": "pb", "what": "choice", "chosen": "wrong, and I say where"},
    {"who": "owner", "row": "pb", "what": "choice", "chosen": "keep this one"},
    {"who": "owner", "row": "pb", "what": "words", "words": "perfect! lock it in."},
    {"who": "a session", "row": "pb", "what": "choice", "chosen": "keep this one"},
    {"who": "owner", "row": "pa", "what": "choice", "chosen": "keep this one"},
    {"who": "owner", "row": "pa", "what": "choice", "chosen": "not sure"},
    None,
    {"who": "owner", "row": "v_0000_0049_joined_T", "what": "choice", "chosen": "keep this one"},
    {"who": "owner", "row": "pc", "what": "choice", "chosen": "keep p4's wording as rendered"},
    {"who": "owner", "row": "kitchen_0604_1009_joined", "what": "choice", "chosen": "keep this one"},
]
#: A line of the verdicts record taken down from the lead's terminal: no `who`, `what: words`, the keep in `chosen`.
#: Lines 2 and 4 are older answers appended late, as the record holds them: the last choice is the latest `at`, not the last line.
VERDICTS = [{"schema": "h3_verdicts/1", "at": "2026-10-16T22:25:00Z", "row": "pc", "what": "words", "chosen": "keep this one",
             "words": "keep it", "source": "the lead's terminal"},
            {"schema": "h3_verdicts/1", "at": "2026-10-14T17:57:00Z", "row": "pc", "what": "choice", "chosen": "not sure",
             "source": "owner_answers.jsonl:1189"},
            {"schema": "h3_verdicts/1", "at": "2026-10-18T09:00:00Z", "row": "pd", "what": "choice", "chosen": "wrong, and I say where",
             "source": "owner_answers.jsonl:1300"},
            {"schema": "h3_verdicts/1", "at": "2026-10-15T09:00:00Z", "row": "pd", "what": "choice", "chosen": "keep this one",
             "source": "owner_answers.jsonl:1200"}]


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(LAID), *map(str, args)], capture_output=True, text=True)


def made(root: Path) -> dict:
    """A share with five renders and a list of five lines: pb on 10-19 for two subjects, its line for `a` with history worth keeping."""
    share, data = root / "share", root / "data" / "v"
    data.mkdir(parents=True)
    for name in ("pa", "pb", "pb2", "pc", "pd"):
        f = share / "Video" / "v" / name / f"{name}_00001.mp4"
        f.parent.mkdir(parents=True)
        f.write_bytes(name.encode() * 64)
    laid = data / "laid.jsonl"
    laid.write_text("")
    base = ["--laid", laid, "--share", share, "--no-rebuild", "a check's batch", "--owner", "unconfirmed"]
    for load, file, shot, subject, extra in (("pa", "pa", "0-9", "a", []),
                                             ("pb", "pb", "10-19", "a", ["--added", "2026-10-20T19:16:05Z", "--added-by", "maker"]),
                                             ("pb", "pb2", "10-19", "b", []), ("pc", "pc", "20-29", "a", []), ("pd", "pd", "30-39", "a", [])):
        r = run("lay", *base, "--shot", shot, "--subject", subject, "--load", load, "--file", f"Video/v/{file}/{file}_00001.mp4",
                "--by", f"laid {load}", *extra)
        assert r.returncode == 0, r.stdout + r.stderr
    lines = [json.loads(x) for x in laid.read_text().splitlines()]
    lines[1]["replaces"] = "pb_old"
    lines[1]["frames"] = [[11, 19]]
    laid.write_text("".join(json.dumps(x) + "\n" for x in lines))
    answers, verdicts = data / "owner_answers.jsonl", data / "verdicts.jsonl"
    answers.write_text("".join(("" if x is None else json.dumps(x)) + "\n" for x in ANSWERS))
    verdicts.write_text("".join(json.dumps(x) + "\n" for x in VERDICTS))
    return {"laid": laid, "share": share, "answers": answers, "verdicts": verdicts,
            "set": ["set-owner", "--laid", laid, "--share", share, "--session", "a check", "--no-rebuild", "a check's set"]}


def owner_is_always_said() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        fx = made(Path(tmp))
        before = fx["laid"].read_bytes()
        common = ["--laid", fx["laid"], "--share", fx["share"], "--subject", "a", "--file", "Video/v/pa/pa_00001.mp4",
                  "--by", "a reason", "--no-rebuild", "a check's batch"]
        refused(run("lay", "--shot", "40-49", "--load", "pe", *common), "the following arguments are required: --owner")
        refused(run("replace", "--shot", "0-9", "--load", "pa", "--replaces", "pa", *common), "the following arguments are required: --owner")
        refused(run("lay", "--shot", "40-49", "--load", "pe", *common, "--owner", "kept"), "--owner-word")
        refused(run("replace", "--shot", "0-9", "--load", "pa", "--replaces", "pa", *common, "--owner", "kept"), "--owner-word")
        refused(run("lay", "--shot", "40-49", "--load", "pb", *common, "--owner", "kept", "--owner-word", f"{fx['answers']}:8"),
                "keep of 'v_0000_0049_joined_T'")
        assert fx["laid"].read_bytes() == before, "a refused lay wrote the list"
        r = run("lay", "--shot", "40-49", "--load", "pb", *common, "--owner", "kept", "--owner-word", f"{fx['answers']}:2")
        assert r.returncode == 0, r.stdout + r.stderr
        last = json.loads(fx["laid"].read_text().splitlines()[-1])
        assert (last["owner"], last["owner_word"]) == ("kept", f"{fx['answers']}:2"), last
    return "lay and replace refuse no --owner and an uncited kept, leaving the list; a cited keep laid with its citation"


def set_owner_changes_only_owner() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        fx = made(Path(tmp))
        old = fx["laid"].read_text().splitlines()
        r = run(*fx["set"], "--load", "pb", "--subject", "a", "--owner", "kept", "--by", "the owner's keep, owner_answers.jsonl:2",
                "--owner-word", f"{fx['answers']}:2")
        assert r.returncode == 0, r.stdout + r.stderr
        new = fx["laid"].read_text().splitlines()
        assert len(new) == 5 and [new[i] for i in (0, 2, 3, 4)] == [old[i] for i in (0, 2, 3, 4)], "a line other than pb's for a changed, or the order did"
        was, now = json.loads(old[1]), json.loads(new[1])
        changed = {k for k in set(was) | set(now) if was.get(k) != now.get(k)}
        assert changed <= {"owner", "by", "owner_word", "history"}, f"set-owner changed {sorted(changed)}"
        assert (now["owner"], now["owner_word"], now["replaces"], now["frames"], now["added"], now["no_rebuild"]) == (
            "kept", f"{fx['answers']}:2", "pb_old", [[11, 19]], "2026-10-20T19:16:05Z", "a check's batch"), now
        h = now["history"]
        assert len(h) == 1 and (h[0]["owner_was"], h[0]["by_was"], h[0]["owner"], h[0]["by_session"], h[0]["no_rebuild"]) == (
            "unconfirmed", "laid pb", "kept", "a check", "a check's set"), h
    return "pb kept in place; replaces, frames, added, no_rebuild and the other four lines unchanged; what it said before in its history"


def kept_needs_the_owners_keep() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        fx = made(Path(tmp))
        before = fx["laid"].read_bytes()
        a = fx["answers"]
        kept = [*fx["set"], "--load", "pb", "--subject", "a", "--owner", "kept", "--by", "a reason"]
        said = []
        for args, words in (([], "--owner-word"),
                            (["--owner-word", f"{a}:4"], "not the owner's line"),
                            (["--owner-word", f"{a}:3"], "not a keep"),
                            (["--owner-word", f"{a}:1"], "not a keep"),
                            (["--owner-word", f"{a}:8"], "keep of 'v_0000_0049_joined_T'"),
                            (["--owner-word", f"{a}:7"], "is blank"),
                            (["--owner-word", f"{a}:99"], "has no line 99"),
                            (["--owner-word", f"{a}"], "FILE:LINE")):
            said.append(refused(run(*kept, *args), words))
        refused(run(*fx["set"], "--load", "pa", "--owner", "kept", "--by", "x", "--owner-word", f"{a}:5"), "not the owner's last choice", "line 6")
        refused(run(*fx["set"], "--load", "pc", "--owner", "kept", "--by", "x", "--owner-word", f"{a}:9"), "not a keep")
        refused(run(*fx["set"], "--load", "pd", "--owner", "kept", "--by", "x", "--owner-word", f"{fx['verdicts']}:4"),
                "not the owner's last choice", "line 3")
        refused(run(*fx["set"], "--load", "pc", "--owner", "kept in a joined clip", "--by", "x", "--owner-word", f"{a}:2"), "not a joined clip")
        refused(run(*fx["set"], "--load", "pb", "--subject", "a", "--owner", "unconfirmed", "--by", "x", "--owner-word", f"{a}:2"), "no keep")
        refused(run(*fx["set"], "--load", "pb", "--subject", "a", "--owner", "unconfirmed", "--by", "x"), "already")
        refused(run(*fx["set"], "--load", "pb", "--owner", "unconfirmed", "--by", "x"), "2 lines", "--subject")
        refused(run(*fx["set"], "--load", "nobody", "--owner", "unconfirmed", "--by", "x"), "0 lines")
        assert fx["laid"].read_bytes() == before, "a refused set-owner wrote the list"
    return f"{len(said) + 8} refusals, each in the tool's words, the list's bytes unchanged"


def history_grows() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        fx = made(Path(tmp))
        a, v = fx["answers"], fx["verdicts"]
        for args in (["--load", "pb", "--subject", "a", "--owner", "kept", "--by", "keep", "--owner-word", f"{a}:2"],
                     ["--load", "pb", "--subject", "a", "--owner", "unconfirmed", "--by", "reset: not the owner's word"],
                     ["--load", "pc", "--owner", "kept", "--by", "kept in the lead's terminal", "--owner-word", f"{v}:1"],
                     ["--load", "pd", "--owner", "kept in a joined clip", "--by", "kept on the joined clip", "--owner-word", f"{a}:10"]):
            r = run(*fx["set"], *args)
            assert r.returncode == 0, r.stdout + r.stderr
        lines = [json.loads(x) for x in fx["laid"].read_text().splitlines()]
        pb, pc, pd = lines[1], lines[3], lines[4]
        assert pb["owner"] == "unconfirmed" and "owner_word" not in pb and [h["owner"] for h in pb["history"]] == ["kept", "unconfirmed"], pb
        assert pb["history"][0]["owner_word"] == f"{a}:2" and pb["history"][1]["by_was"] == "keep", pb["history"]
        assert (pc["owner"], pc["owner_word"]) == ("kept", f"{v}:1"), pc
        assert (pd["owner"], pd["owner_word"]) == ("kept in a joined clip", f"{a}:10"), pd
    return "kept then reset: two entries, the citation kept in the history; a verdicts-form keep and a joined clip's keep taken"


def a_removed_line_leaves_its_record() -> str:
    """`remove` takes `--by` and keeps the line whole, with why, in `laid.removed.jsonl`; with `--no-rebuild` no other
    line of the list changes (until 2026-10-21 its reason was written onto the last line left)."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = made(Path(tmp))
        before = fx["laid"].read_bytes()
        # a reason no fixture line carries, so a write of it onto another line shows
        base = ["remove", "--laid", fx["laid"], "--share", fx["share"], "--load", "pc", "--no-rebuild", "a check's removal"]
        refused(run(*base), "--by")
        refused(run(*base, "--by", "   "), "--by is required")
        assert fx["laid"].read_bytes() == before and not (fx["laid"].parent / "laid.removed.jsonl").exists(), "a refused remove wrote"
        # --frames outside the shot is refused by the list's reader before anything is written
        refused(run("lay", "--laid", fx["laid"], "--share", fx["share"], "--shot", "40-49", "--subject", "a", "--load", "pe",
                    "--file", "Video/v/pa/pa_00001.mp4", "--owner", "unconfirmed", "--by", "x", "--frames", "35-45",
                    "--no-rebuild", "a check's batch"), "are not [first, last] spans inside the shot 40-49")
        assert fx["laid"].read_bytes() == before, "a refused lay replaced the list"
        old = fx["laid"].read_text().splitlines()
        r = run(*base, "--by", "the owner's look says it goes")
        assert r.returncode == 0, r.stdout + r.stderr
        new = fx["laid"].read_text().splitlines()
        assert new == [old[0], old[1], old[2], old[4]], "remove changed a line it did not take out"
        rec = [json.loads(x) for x in (fx["laid"].parent / "laid.removed.jsonl").read_text().splitlines()]
        assert len(rec) == 1 and rec[0]["removed"] == json.loads(old[3]) and rec[0]["by"] == "the owner's look says it goes", rec
        assert rec[0]["no_rebuild"] == "a check's removal" and rec[0]["laid_sha256_before"] != rec[0]["laid_sha256_after"], rec[0]
        # a remove that matches several lines keeps a record of each
        r = run("remove", "--laid", fx["laid"], "--share", fx["share"], "--load", "pb", "--by", "both go", "--no-rebuild", "a check's removal")
        assert r.returncode == 0, r.stdout + r.stderr
        rec = [json.loads(x) for x in (fx["laid"].parent / "laid.removed.jsonl").read_text().splitlines()]
        assert [(x["removed"]["load"], x["removed"]["subject"], x["by"]) for x in rec[1:]] == [("pb", "a", "both go"), ("pb", "b", "both go")], rec
        assert len(fx["laid"].read_text().splitlines()) == 2, fx["laid"].read_text()
    return "no --by refused and nothing written; the line kept whole with why in laid.removed.jsonl; no other line touched"


def keeps_verifies_and_refuses() -> str:
    """`keeps` verifies disjoint frame spans on one shot (`1466-1534` and `1575-1729`), classifies superseded keeps on
    overlapping frames, and refuses (`KEEP_NOT_LAID`, `LAID_MARKED_WRONG`) with no traceback."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "data" / "v"
        data.mkdir(parents=True)
        laid_path = data / "laid.jsonl"
        answers_path = data / "owner_answers.jsonl"
        verdicts_path = data / "verdicts.jsonl"
        out_json = data / "keeps.json"

        laid_rows = [
            {
                "schema": "h3_laid/1",
                "first": 1466,
                "last": 1729,
                "subject": "lead",
                "load": "lead_fp_1466b",
                "file": "Video/v/lead_fp_1466b/lead_fp_1466b_00001.mp4",
                "sha256": "a" * 64,
                "owner": "kept",
                "by": "the owner's keep of 2026-10-16T10:00:00Z",
                "frames": [[1466, 1534]],
                "added": "2026-10-16T12:00:00Z",
                "added_by": "check",
            },
            {
                "schema": "h3_laid/1",
                "first": 1466,
                "last": 1729,
                "subject": "lead",
                "load": "lead_fp_1575b",
                "file": "Video/v/lead_fp_1575b/lead_fp_1575b_00001.mp4",
                "sha256": "b" * 64,
                "owner": "kept",
                "by": "the owner's keep of 2026-10-16T10:05:00Z",
                "frames": [[1575, 1729]],
                "added": "2026-10-16T12:00:00Z",
                "added_by": "check",
            },
            {
                "schema": "h3_laid/1",
                "first": 1466,
                "last": 1729,
                "subject": "third",
                "load": "third_fp_1466b",
                "file": "Video/v/third_fp_1466b/third_fp_1466b_00001.mp4",
                "sha256": "c" * 64,
                "owner": "kept",
                "by": "the owner's keep of 2026-10-16T11:00:00Z",
                "frames": [[1466, 1531]],
                "added": "2026-10-16T12:00:00Z",
                "added_by": "check",
                "replaces": "third_fp_1466a",
            },
        ]
        answers = [
            {"at": "2026-10-15T09:00:00Z", "who": "owner", "row": "lead_fp_1466a", "frames": [1466, 1534], "what": "choice", "chosen": "keep this one"},
            {"at": "2026-10-16T10:00:00Z", "who": "owner", "row": "lead_fp_1466b", "frames": [1466, 1534], "what": "choice", "chosen": "keep this one"},
            {"at": "2026-10-16T10:05:00Z", "who": "owner", "row": "lead_fp_1575b", "frames": [1575, 1729], "what": "choice", "chosen": "keep this one"},
            {"at": "2026-10-14T08:00:00Z", "who": "owner", "row": "third_fp_1466a", "frames": [1466, 1531], "what": "choice", "chosen": "keep this one"},
            {"at": "2026-10-16T11:00:00Z", "who": "owner", "row": "third_fp_1466b", "frames": [1466, 1531], "what": "choice", "chosen": "keep this one"},
        ]
        laid_path.write_text("".join(json.dumps(x) + "\n" for x in laid_rows))
        answers_path.write_text("".join(json.dumps(x) + "\n" for x in answers))
        verdicts_path.write_text("")

        # 1. Disjoint spans on one shot (1466-1534 and 1575-1729) + superseded keeps pass with exit 0
        r = run("keeps", "--video", "v", "--laid", laid_path, "--answers", answers_path, "--verdicts", verdicts_path, "--json", out_json)
        assert r.returncode == 0, r.stdout + r.stderr
        assert (
            "3 laid rows across 1 shots (1 multi-subject composite shots, 1 split-span subject shots); "
            "5 kept loads verified; 2 historical keeps superseded; 0 unreplaced removals; 0 newer unlaid renders; "
            "0 kept loads missing from laid.jsonl; 0 laid loads marked wrong"
        ) in r.stdout, r.stdout
        rep = json.loads(out_json.read_text())
        assert [s["load"] for s in rep["superseded_keeps"]] == ["lead_fp_1466a", "third_fp_1466a"], rep["superseded_keeps"]

        # 2. Removing the second disjoint span (1575-1729) while lead_fp_1466b (1466-1534) stays laid refuses as KEEP_NOT_LAID
        laid_path.write_text("".join(json.dumps(x) + "\n" for x in [laid_rows[0], laid_rows[2]]))
        refused(
            run("keeps", "--video", "v", "--laid", laid_path, "--answers", answers_path, "--verdicts", verdicts_path),
            "KEEP_NOT_LAID",
            "lead_fp_1575b",
            "1575-1729",
        )

        # 3. Restoring laid_rows and marking a laid load wrong later refuses as LAID_MARKED_WRONG
        laid_path.write_text("".join(json.dumps(x) + "\n" for x in laid_rows))
        answers_wrong = [*answers, {"at": "2026-10-17T12:00:00Z", "who": "owner", "row": "third_fp_1466b", "frames": [1466, 1531], "what": "choice", "chosen": "wrong, and I say where"}]
        answers_path.write_text("".join(json.dumps(x) + "\n" for x in answers_wrong))
        refused(
            run("keeps", "--video", "v", "--laid", laid_path, "--answers", answers_path, "--verdicts", verdicts_path),
            "LAID_MARKED_WRONG",
            "third_fp_1466b",
            "wrong, and I say where",
        )
    return "split-span and multi-subject shot verified; superseded keeps reported; KEEP_NOT_LAID and LAID_MARKED_WRONG refused in the tool's words"


def partial_lay_requires_coverage_or_override() -> str:
    """Gate 3: laying `--frames` narrower than `--shot FIRST-LAST` refuses when frames of `FIRST..LAST` remain
    uncovered by either another laid row on that shot/subject or a `not seen` look in `shots.json`, unless
    `--allow-partial-lay WHY` is given (recording `partial_lay_why` on the line)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        share = root / "share"
        share.mkdir(parents=True)
        (share / "p1_00001.mp4").write_bytes(b"piece 1")
        (share / "p2_00001.mp4").write_bytes(b"piece 2")
        vdir = root / "data" / "v"
        vdir.mkdir(parents=True)
        laid_path = vdir / "laid.jsonl"
        shots_path = vdir / "shots.json"
        shots_path.write_text(
            json.dumps(
                {
                    "shots": [
                        {
                            "shot": 1,
                            "first": 100,
                            "last": 199,
                            "who": {"lead": {"seen_looks": []}},
                        }
                    ]
                }
            )
        )

        # 1. Partial lay 100-149 on shot 100-199 without coverage or --allow-partial-lay refuses naming 150-199
        refused(
            run(
                "lay",
                "--laid",
                laid_path,
                "--share",
                share,
                "--shot",
                "100-199",
                "--subject",
                "lead",
                "--load",
                "lead_fp_100a",
                "--file",
                "p1_00001.mp4",
                "--frames",
                "100-149",
                "--owner",
                "unconfirmed",
                "--by",
                "first half",
                "--no-rebuild",
                "test",
            ),
            "partial lay",
            "150-199",
            "--allow-partial-lay",
        )
        assert not laid_path.exists()

        # 2. Empty --allow-partial-lay refuses
        refused(
            run(
                "lay",
                "--laid",
                laid_path,
                "--share",
                share,
                "--shot",
                "100-199",
                "--subject",
                "lead",
                "--load",
                "lead_fp_100a",
                "--file",
                "p1_00001.mp4",
                "--frames",
                "100-149",
                "--owner",
                "unconfirmed",
                "--by",
                "first half",
                "--allow-partial-lay",
                "--no-rebuild",
                "test",
            ),
            "--allow-partial-lay requires a non-empty reason (WHY)",
        )

        # 3. With --allow-partial-lay WHY, the first half is laid and partial_lay_why is recorded on the line
        r1 = run(
            "lay",
            "--laid",
            laid_path,
            "--share",
            share,
            "--shot",
            "100-199",
            "--subject",
            "lead",
            "--load",
            "lead_fp_100a",
            "--file",
            "p1_00001.mp4",
            "--frames",
            "100-149",
            "--owner",
            "unconfirmed",
            "--by",
            "first half",
            "--allow-partial-lay",
            "second half 150-199 laid next",
            "--no-rebuild",
            "test",
        )
        assert r1.returncode == 0, r1.stdout + r1.stderr
        rows = [json.loads(line) for line in laid_path.read_text().splitlines() if line.strip()]
        assert len(rows) == 1 and rows[0]["partial_lay_why"] == "second half 150-199 laid next", rows

        # 4. Laying the second half 150-199 completes 100-199 without needing --allow-partial-lay
        r2 = run(
            "lay",
            "--laid",
            laid_path,
            "--share",
            share,
            "--shot",
            "100-199",
            "--subject",
            "lead",
            "--load",
            "lead_fp_150a",
            "--file",
            "p2_00001.mp4",
            "--frames",
            "150-199",
            "--owner",
            "unconfirmed",
            "--by",
            "second half",
            "--no-rebuild",
            "test",
        )
        assert r2.returncode == 0, r2.stdout + r2.stderr

        # 5. With plan.json parts ["face_and_neck", "hair"], if only face_and_neck is `not seen` on 150-199 (real first/last schema),
        #    150-199 is still uncovered (hair is still owed); once hair is also `not seen`, laying 100-149 passes; and a later `seen`
        #    by `at` on 180-199 makes 180-199 uncovered again.
        laid_path.unlink()
        (vdir / "plan.json").write_text(
            json.dumps({"subjects": {"lead": {"parts": ["face_and_neck", "hair"]}}})
        )
        shots_path.write_text(
            json.dumps(
                {
                    "shots": [
                        {
                            "shot": 1,
                            "first": 100,
                            "last": 199,
                            "who": {
                                "lead": {
                                    "seen_looks": [
                                        {
                                            "first": 150,
                                            "last": 199,
                                            "is": "not seen",
                                            "part": "face_and_neck",
                                            "by": "reader",
                                            "at": "2026-10-25T10:00:00Z",
                                        }
                                    ]
                                }
                            },
                        }
                    ]
                }
            )
        )
        refused(
            run(
                "lay",
                "--laid",
                laid_path,
                "--share",
                share,
                "--shot",
                "100-199",
                "--subject",
                "lead",
                "--load",
                "lead_fp_100a",
                "--file",
                "p1_00001.mp4",
                "--frames",
                "100-149",
                "--owner",
                "unconfirmed",
                "--by",
                "only one part not seen",
                "--no-rebuild",
                "test",
            ),
            "partial lay",
            "150-199",
        )

        # Adding hair `not seen` (or a later `seen` on 180-199 with an earlier `not seen` ordered out-of-order in JSON)
        shots_path.write_text(
            json.dumps(
                {
                    "shots": [
                        {
                            "shot": 1,
                            "first": 100,
                            "last": 199,
                            "who": {
                                "lead": {
                                    "seen_looks": [
                                        {
                                            "first": 180,
                                            "last": 199,
                                            "is": "seen",
                                            "part": "hair",
                                            "by": "reader",
                                            "at": "2026-10-25T11:00:00Z",
                                        },
                                        {
                                            "first": 150,
                                            "last": 199,
                                            "is": "not seen",
                                            "part": "head",
                                            "by": "reader",
                                            "at": "2026-10-25T10:00:00Z",
                                        },
                                    ]
                                }
                            },
                        }
                    ]
                }
            )
        )
        refused(
            run(
                "lay",
                "--laid",
                laid_path,
                "--share",
                share,
                "--shot",
                "100-199",
                "--subject",
                "lead",
                "--load",
                "lead_fp_100a",
                "--file",
                "p1_00001.mp4",
                "--frames",
                "100-149",
                "--owner",
                "unconfirmed",
                "--by",
                "later seen on 180-199",
                "--no-rebuild",
                "test",
            ),
            "partial lay",
            "180-199",
        )

        shots_path.write_text(
            json.dumps(
                {
                    "shots": [
                        {
                            "shot": 1,
                            "first": 100,
                            "last": 199,
                            "who": {
                                "lead": {
                                    "seen_looks": [
                                        {
                                            "first": 150,
                                            "last": 199,
                                            "is": "not seen",
                                            "part": "face_and_neck",
                                            "by": "reader",
                                            "at": "2026-10-25T10:00:00Z",
                                        },
                                        {
                                            "first": 150,
                                            "last": 199,
                                            "is": "not seen",
                                            "part": "hair",
                                            "by": "reader",
                                            "at": "2026-10-25T10:05:00Z",
                                        },
                                    ]
                                }
                            },
                        }
                    ]
                }
            )
        )
        r3 = run(
            "lay",
            "--laid",
            laid_path,
            "--share",
            share,
            "--shot",
            "100-199",
            "--subject",
            "lead",
            "--load",
            "lead_fp_100a",
            "--file",
            "p1_00001.mp4",
            "--frames",
            "100-149",
            "--owner",
            "unconfirmed",
            "--by",
            "first half with 150-199 not seen across all parts",
            "--no-rebuild",
            "test",
        )
        assert r3.returncode == 0, r3.stdout + r3.stderr

        # Parallel Pass-1 guard: laying a second subject whose job.source is raw (missing lead's laid piece) is refused
        (share / "p_sec_00001.mp4").write_bytes(b"p_sec" * 32)
        job_dir = laid_path.parent.parent / "2026-10-25_second_hfp_100a_job"
        job_dir.mkdir(parents=True)
        (job_dir / "job.render.json").write_text(json.dumps({"job": {"source": "source.mp4"}}))
        refused(
            run(
                "lay",
                "--laid",
                laid_path,
                "--share",
                share,
                "--shot",
                "100-199",
                "--subject",
                "second",
                "--load",
                "second_hfp_100a",
                "--file",
                "p_sec_00001.mp4",
                "--frames",
                "100-149",
                "--owner",
                "unconfirmed",
                "--by",
                "parallel raw pass",
                "--allow-partial-lay",
                "150-199 not in frame",
                "--no-rebuild",
                "test",
            ),
            "was rendered on source.mp4 which does not hold lead's laid piece lead_fp_100a on 100-149",
            "--allow-parallel-raw WHY",
        )
        # If job.source is a cut whose sidecar holds lead on 100-149, but from a stale load (lead_fp_100_old != lead_fp_100a), it is also refused
        cut_mp4 = share / "v_0000_0199_joined_2026-10-25T1000Z_from_laid.mp4"
        cut_sc = Path(f"{cut_mp4}.json")
        cut_sc.write_text(
            json.dumps(
                {
                    "pieces": [
                        {
                            "first": 100,
                            "last": 149,
                            "subject": "lead",
                            "name": "lead_fp_100_old",
                            "load": {"load": "lead_fp_100_old", "stage": "render"},
                        }
                    ]
                }
            )
        )
        (job_dir / "job.render.json").write_text(json.dumps({"job": {"source": cut_mp4.name}}))
        refused(
            run(
                "lay",
                "--laid",
                laid_path,
                "--share",
                share,
                "--shot",
                "100-199",
                "--subject",
                "second",
                "--load",
                "second_hfp_100a",
                "--file",
                "p_sec_00001.mp4",
                "--frames",
                "100-149",
                "--owner",
                "unconfirmed",
                "--by",
                "stale pass-1 cut",
                "--allow-partial-lay",
                "150-199 not in frame",
                "--no-rebuild",
                "test",
            ),
            "does not hold lead's laid piece lead_fp_100a on 100-149",
        )
        # When the cut sidecar holds lead's actual laid piece lead_fp_100a on 100-149, Pass 2 lay succeeds without --allow-parallel-raw
        cut_sc.write_text(
            json.dumps(
                {
                    "pieces": [
                        {
                            "first": 100,
                            "last": 149,
                            "subject": "lead",
                            "name": "lead_fp_100a",
                            "load": {"load": "lead_fp_100a", "stage": "render"},
                        }
                    ]
                }
            )
        )
        r4 = run(
            "lay",
            "--laid",
            laid_path,
            "--share",
            share,
            "--shot",
            "100-199",
            "--subject",
            "second",
            "--load",
            "second_hfp_100a",
            "--file",
            "p_sec_00001.mp4",
            "--frames",
            "100-149",
            "--owner",
            "unconfirmed",
            "--by",
            "pass 2 on cut holding lead_fp_100a",
            "--allow-partial-lay",
            "150-199 not in frame",
            "--no-rebuild",
            "test",
        )
        assert r4.returncode == 0, r4.stdout + r4.stderr
    return "partial lay leaving uncovered shot frames and parallel/stale Pass-1 raw lay refused; passes with cut holding exact laid load, sibling row, `not seen`, or override"


def same_subject_carve_and_restore() -> str:
    """When a newer keep overlaps an earlier row of the same subject, it carves the overlapping frames (`carved_by`),
    moves a wholly carved row to `laid.removed.jsonl` (`carved_whole_by`, populating `replaces` on same-shot whole carve),
    and restores carved frames / carved-whole rows when the newer row is removed."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        share = root / "share"
        share.mkdir(parents=True)
        for name in ("a1", "a2", "a3"):
            (share / f"{name}_00001.mp4").write_bytes(name.encode() * 32)
        vdir = root / "data" / "v"
        vdir.mkdir(parents=True)
        # Create capture manifests for lead_fp_100a (replaces all of 100-199) and lead_fp_150b (leaves 180-189 as original)
        cap_a = root / "data" / "2026-10-20_lead_fp_100a"
        cap_a.mkdir(parents=True)
        (cap_a / "manifest.json").write_text(
            json.dumps({"first_frame": 100, "frames": 100, "subjects": [{"label": "lead", "sightings": [{"parts_emptied_on_source_frames": []}]}]})
        )
        cap_b = root / "data" / "2026-10-21_lead_fp_150b"
        cap_b.mkdir(parents=True)
        (cap_b / "manifest.json").write_text(
            json.dumps({"first_frame": 150, "frames": 50, "subjects": [{"label": "lead", "sightings": [{"parts_emptied_on_source_frames": [[180, 189]]}]}]})
        )
        laid_path = vdir / "laid.jsonl"
        answers_path = vdir / "owner_answers.jsonl"
        verdicts_path = vdir / "verdicts.jsonl"
        verdicts_path.write_text("")
        answers = [
            {"at": "2026-10-20T10:00:00Z", "who": "owner", "row": "lead_fp_100a", "frames": [100, 199], "what": "choice", "chosen": "keep this one"},
            {"at": "2026-10-21T10:00:00Z", "who": "owner", "row": "lead_fp_150b", "frames": [150, 199], "what": "choice", "chosen": "keep this one"},
            {"at": "2026-10-22T10:00:00Z", "who": "owner", "row": "lead_fp_100c", "frames": [100, 199], "what": "choice", "chosen": "keep this one"},
        ]
        answers_path.write_text("".join(json.dumps(x) + "\n" for x in answers))

        # 1. Lay lead_fp_100a (kept at 2026-10-20T10:00:00Z) on 100-199
        r1 = run(
            "lay", "--laid", laid_path, "--share", share, "--shot", "100-199", "--subject", "lead",
            "--load", "lead_fp_100a", "--file", "a1_00001.mp4", "--owner", "kept",
            "--owner-word", f"{answers_path}:1", "--by", "first keep", "--no-rebuild", "test",
        )
        assert r1.returncode == 0, r1.stdout + r1.stderr

        # 2. Unconfirmed row cannot carve an active keep
        refused(
            run(
                "lay", "--laid", laid_path, "--share", share, "--shot", "100-199", "--subject", "lead",
                "--load", "lead_fp_150u", "--file", "a2_00001.mp4", "--frames", "150-199", "--owner", "unconfirmed",
                "--by", "unconfirmed try", "--no-rebuild", "test",
            ),
            "cannot carve or replace active keep lead_fp_100a",
        )

        # 3. Newer keep lead_fp_150b on 150-199 carves lead_fp_100a to 100-149 and records carved_by
        r2 = run(
            "lay", "--laid", laid_path, "--share", share, "--shot", "100-199", "--subject", "lead",
            "--load", "lead_fp_150b", "--file", "a2_00001.mp4", "--frames", "150-199", "--owner", "kept",
            "--owner-word", f"{answers_path}:2", "--by", "second keep carves tail", "--no-rebuild", "test",
        )
        assert r2.returncode == 0, r2.stdout + r2.stderr
        rows = [json.loads(line) for line in laid_path.read_text().splitlines() if line.strip()]
        assert len(rows) == 2, rows
        assert rows[0]["load"] == "lead_fp_100a" and rows[0]["frames"] == [[100, 149]], rows[0]
        assert rows[0]["carved_by"][0]["load"] == "lead_fp_150b" and rows[0]["carved_by"][0]["frames"] == [[150, 199]], rows[0]
        assert rows[1]["load"] == "lead_fp_150b" and rows[1]["frames"] == [[150, 199]], rows[1]
        assert rows[1]["original_restored_over_replacement"] == [
            {
                "older_load": "lead_fp_100a",
                "frames": [[180, 189]],
                "why": "newer keep leaves frame(s) as original where older keep replaced subject",
            }
        ], rows[1]

        # 4. Newer keep lead_fp_100c on 100-199 carves both whole, moves them to laid.removed.jsonl with carved_whole_by, and records replaces
        r3 = run(
            "lay", "--laid", laid_path, "--share", share, "--shot", "100-199", "--subject", "lead",
            "--load", "lead_fp_100c", "--file", "a3_00001.mp4", "--frames", "100-199", "--owner", "kept",
            "--owner-word", f"{answers_path}:3", "--by", "third keep carves whole shot", "--no-rebuild", "test",
        )
        assert r3.returncode == 0, r3.stdout + r3.stderr
        rows = [json.loads(line) for line in laid_path.read_text().splitlines() if line.strip()]
        assert len(rows) == 1 and rows[0]["load"] == "lead_fp_100c" and rows[0]["frames"] == [[100, 199]], rows
        assert rows[0].get("replaces") == ["lead_fp_100a", "lead_fp_150b"], rows[0]

        # 5. Removing lead_fp_100c restores lead_fp_100a (100-149) and lead_fp_150b (150-199); removing lead_fp_150b restores lead_fp_100a to 100-199
        r4 = run(
            "remove", "--laid", laid_path, "--share", share, "--load", "lead_fp_100c",
            "--by", "undo third keep", "--no-rebuild", "test",
        )
        assert r4.returncode == 0, r4.stdout + r4.stderr
        rows = [json.loads(line) for line in laid_path.read_text().splitlines() if line.strip()]
        assert [(x["load"], x["frames"]) for x in rows] == [
            ("lead_fp_100a", [[100, 149]]),
            ("lead_fp_150b", [[150, 199]]),
        ], rows
        r5 = run(
            "remove", "--laid", laid_path, "--share", share, "--load", "lead_fp_150b",
            "--by", "undo second keep", "--no-rebuild", "test",
        )
        assert r5.returncode == 0, r5.stdout + r5.stderr
        rows = [json.loads(line) for line in laid_path.read_text().splitlines() if line.strip()]
        assert len(rows) == 1 and rows[0]["load"] == "lead_fp_100a" and rows[0]["frames"] == [[100, 199]] and "carved_by" not in rows[0], rows
    return "same-subject keep carves overlapping frames, records carved_by/carved_whole_by/replaces, and remove restores them"


def sync_and_keeps_rules() -> str:
    """`sync` and `keeps` preserve existing partial lays and `replaces`, subtract `not seen` frames on unlaid keeps,
    respect `replaces` superseding, never use a load-letter heuristic, and never rewrite already-kept rows on path differences."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        share = root / "share"
        v_out = share / "Video" / "v"
        v_out.mkdir(parents=True)
        for ld in ("lead_fp_100d", "lead_fp_100b", "lead_fp_200c", "lead_fp_200d", "lead_fp_300a"):
            p = v_out / f"{ld}_00001.mp4"
            p.write_bytes(ld.encode() * 32)
        vdir = root / "data" / "v"
        vdir.mkdir(parents=True)
        laid_path = vdir / "laid.jsonl"
        answers_path = vdir / "owner_answers.jsonl"
        verdicts_path = vdir / "verdicts.jsonl"
        verdicts_path.write_text("")
        shots_path = vdir / "shots.json"
        shots_path.write_text(
            json.dumps(
                {
                    "shots": [
                        {"shot": 1, "first": 100, "last": 199, "who": {"lead": {"seen_looks": []}}},
                        {
                            "shot": 2,
                            "first": 200,
                            "last": 299,
                            "who": {
                                "lead": {
                                    "seen_looks": [
                                        {
                                            "first": 290,
                                            "last": 299,
                                            "is": "not seen",
                                            "part": "head",
                                            "by": "reader",
                                            "at": "2026-10-20T00:00:00Z",
                                        }
                                    ]
                                }
                            },
                        },
                        {"shot": 3, "first": 300, "last": 399, "who": {"lead": {"seen_looks": []}}},
                    ]
                }
            )
        )
        # Shot 1 (100-199): lead_fp_100d is older than lead_fp_100b ('b' < 'd' alphabetically, but 'b' is NEWER by timestamp!)
        # Shot 2 (200-299): lead_fp_200d (200-289) is laid with replaces="lead_fp_200c", while 290-299 is `not seen` in shots.json
        # Shot 3 (300-399): lead_fp_300a is intentionally laid on 300-350 (partial lay) with relative owner_word "data/v/owner_answers.jsonl:4"
        answers = [
            {"at": "2026-10-20T10:00:00Z", "who": "owner", "row": "lead_fp_100d", "frames": [100, 199], "what": "choice", "chosen": "keep this one"},
            {"at": "2026-10-25T10:00:00Z", "who": "owner", "row": "lead_fp_100b", "frames": [100, 199], "what": "choice", "chosen": "keep this one"},
            {"at": "2026-10-21T10:00:00Z", "who": "owner", "row": "lead_fp_200c", "frames": [200, 299], "what": "choice", "chosen": "keep this one"},
            {"at": "2026-10-22T10:00:00Z", "who": "owner", "row": "lead_fp_200d", "frames": [200, 289], "what": "choice", "chosen": "keep this one"},
            {"at": "2026-10-23T10:00:00Z", "who": "owner", "row": "lead_fp_300a", "frames": [300, 399], "what": "choice", "chosen": "keep this one"},
        ]
        answers_path.write_text("".join(json.dumps(x) + "\n" for x in answers))

        import hashlib
        def _sha(ld: str) -> str:
            return hashlib.sha256((v_out / f"{ld}_00001.mp4").read_bytes()).hexdigest()

        initial_laid = [
            {
                "schema": "h3_laid/1",
                "first": 100,
                "last": 199,
                "subject": "lead",
                "load": "lead_fp_100d",
                "file": "Video/v/lead_fp_100d_00001.mp4",
                "sha256": _sha("lead_fp_100d"),
                "owner": "kept",
                "by": "older keep",
                "owner_word": "data/v/owner_answers.jsonl:1",
                "frames": [[100, 199]],
                "added": "2026-10-20T10:00:00Z",
                "added_by": "check",
            },
            {
                "schema": "h3_laid/1",
                "first": 200,
                "last": 299,
                "subject": "lead",
                "load": "lead_fp_200d",
                "file": "Video/v/lead_fp_200d_00001.mp4",
                "sha256": _sha("lead_fp_200d"),
                "owner": "kept",
                "by": "kept on 200-289",
                "owner_word": "data/v/owner_answers.jsonl:4",
                "frames": [[200, 289]],
                "added": "2026-10-22T10:00:00Z",
                "added_by": "check",
                "replaces": "lead_fp_200c",
            },
            {
                "schema": "h3_laid/1",
                "first": 300,
                "last": 399,
                "subject": "lead",
                "load": "lead_fp_300a",
                "file": "Video/v/lead_fp_300a_00001.mp4",
                "sha256": _sha("lead_fp_300a"),
                "owner": "kept",
                "by": "kept partial lay on 300-350",
                "owner_word": "data/v/owner_answers.jsonl:5",
                "frames": [[300, 350]],
                "partial_lay_why": "351-399 awaiting decision",
                "added": "2026-10-23T10:00:00Z",
                "added_by": "check",
            },
        ]
        laid_path.write_text("".join(json.dumps(x) + "\n" for x in initial_laid))

        # Before sync: `keeps` must NOT treat lead_fp_100b (newer timestamp, earlier letter 'b' < 'd') as superseded by lead_fp_100d!
        refused(
            run("keeps", "--video", "v", "--laid", laid_path, "--answers", answers_path, "--verdicts", verdicts_path),
            "KEEP_NOT_LAID",
            "lead_fp_100b",
        )

        # Run `sync`:
        # - lead_fp_100b carves whole lead_fp_100d and populates replaces="lead_fp_100d"
        # - lead_fp_200c is NOT laid on 290-299 (excluded by both replaces on lead_fp_200d and not seen in shots.json)
        # - lead_fp_200d preserves replaces="lead_fp_200c" and is not rewritten
        # - lead_fp_300a preserves frames=[[300, 350]] and its by/owner_word/history are NOT rewritten
        r_sync = run(
            "sync", "--video", "v", "--laid", laid_path, "--share", share,
            "--answers", answers_path, "--verdicts", verdicts_path, "--data", vdir,
        )
        assert r_sync.returncode == 0, r_sync.stdout + r_sync.stderr
        synced_rows = [json.loads(line) for line in laid_path.read_text().splitlines() if line.strip()]
        assert [x["load"] for x in synced_rows] == ["lead_fp_200d", "lead_fp_300a", "lead_fp_100b"], synced_rows
        assert synced_rows[0]["replaces"] == "lead_fp_200c" and "history" not in synced_rows[0], synced_rows[0]
        assert synced_rows[1]["frames"] == [[300, 350]] and synced_rows[1]["by"] == "kept partial lay on 300-350" and "history" not in synced_rows[1], synced_rows[1]
        assert synced_rows[2]["replaces"] == "lead_fp_100d" and synced_rows[2]["frames"] == [[100, 199]], synced_rows[2]

        # Re-running `sync` is a clean no-op
        before_resync = laid_path.read_bytes()
        r_sync2 = run(
            "sync", "--video", "v", "--laid", laid_path, "--share", share,
            "--answers", answers_path, "--verdicts", verdicts_path, "--data", vdir,
        )
        assert r_sync2.returncode == 0 and "already in sync" in r_sync2.stdout, r_sync2.stdout + r_sync2.stderr
        assert laid_path.read_bytes() == before_resync

        # Also verify Rule 2(a) alone (replaces without `not seen` in shots.json):
        # If shots.json has no `not seen` on 290-299, lead_fp_200c (kept on 200-299 at 2026-10-21) is still excluded from sync and keeps
        # because active keep lead_fp_200d (kept at 2026-10-22) has replaces="lead_fp_200c".
        shots_path.write_text(
            json.dumps(
                {
                    "shots": [
                        {"shot": 1, "first": 100, "last": 199, "who": {"lead": {"seen_looks": []}}},
                        {"shot": 2, "first": 200, "last": 299, "who": {"lead": {"seen_looks": []}}},
                        {"shot": 3, "first": 300, "last": 399, "who": {"lead": {"seen_looks": []}}},
                    ]
                }
            )
        )
        r_sync3 = run(
            "sync", "--video", "v", "--laid", laid_path, "--share", share,
            "--answers", answers_path, "--verdicts", verdicts_path, "--data", vdir,
        )
        assert r_sync3.returncode == 0 and "already in sync" in r_sync3.stdout, r_sync3.stdout + r_sync3.stderr
        r_keeps2 = run("keeps", "--video", "v", "--laid", laid_path, "--answers", answers_path, "--verdicts", verdicts_path)
        assert r_keeps2.returncode == 0, r_keeps2.stdout + r_keeps2.stderr
        assert "0 kept loads missing from laid.jsonl; 0 laid loads marked wrong" in r_keeps2.stdout, r_keeps2.stdout
    return "sync and keeps obey chronological keep order without letter heuristics, preserve partial lays and replaces, and exclude not-seen/replaced keeps"


if __name__ == "__main__":
    case("owner_is_always_said", owner_is_always_said)
    case("set_owner_changes_only_owner", set_owner_changes_only_owner)
    case("kept_needs_the_owners_keep", kept_needs_the_owners_keep)
    case("history_grows", history_grows)
    case("a_removed_line_leaves_its_record", a_removed_line_leaves_its_record)
    case("keeps_verifies_and_refuses", keeps_verifies_and_refuses)
    case("partial_lay_requires_coverage_or_override", partial_lay_requires_coverage_or_override)
    case("same_subject_carve_and_restore", same_subject_carve_and_restore)
    case("sync_and_keeps_rules", sync_and_keeps_rules)
    sys.exit(finish())
