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


if __name__ == "__main__":
    case("owner_is_always_said", owner_is_always_said)
    case("set_owner_changes_only_owner", set_owner_changes_only_owner)
    case("kept_needs_the_owners_keep", kept_needs_the_owners_keep)
    case("history_grows", history_grows)
    case("a_removed_line_leaves_its_record", a_removed_line_leaves_its_record)
    sys.exit(finish())
