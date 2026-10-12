#!/usr/bin/env python3
"""`bench/page_answers_to_verdicts.py` copies the owner's clip verdicts from the page's raw file into the record; each case is a way a line could be lost, doubled, reworded or put on the wrong render.

Why it exists: about fifty of the owner's typed lines on single clips lived in the page's own file for a day
with nothing reading them back beside the render. The copy is only worth having if it is complete, exact, once,
and on the right render.

  a_choice_and_a_words_line_are_both_kept     with the render the item names, its frames, and the line's frame
  nothing_is_reworded                         the words are byte for byte the page's
  a_second_run_writes_nothing                 the same source line is not copied twice
  a_line_on_a_list_row_is_left_to_its_tool    counted, not written
  another_persons_line_is_left_out
  an_item_taken_off_the_page_is_still_known   through a second --items file
  an_empty_submit_is_not_a_verdict            a choice with no button or words with no text writes nothing
  a_view_that_is_not_a_render_is_kept_as_such render null
  show_prints_a_renders_lines_in_order
  dry_run_writes_nothing

    <python> bench/check_page_answers_to_verdicts.py

No model, no card, no imports beyond the standard library.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "page_answers_to_verdicts.py"


def line(n, **kw):
    d = {"at": f"2026-10-12T10:00:{n:02d}Z", "who": "owner", "what": "choice"}
    d.update(kw)
    return d


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True)


def main() -> int:
    fails = 0

    def case(name, ok, why=""):
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {why}"))
        fails += not ok

    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        items = folder / "items.json"
        items.write_text(json.dumps({"items": [
            {"id": "r__x__one", "title": "one", "render": "Video/x/one/one_00001.mp4", "first": 10, "frames": 20},
            {"id": "layers_view", "title": "a view", "first": 10, "frames": 20}]}))
        removed = folder / "removed.json"
        removed.write_text(json.dumps({"items": [{"id": "r__x__gone", "title": "gone", "render": "Video/x/gone/gone_00001.mp4", "first": 30, "frames": 5}]}))
        answers = folder / "answers.jsonl"
        answers.write_text("".join(json.dumps(x) + "\n" for x in [
            line(1, row="r__x__one", chosen="fine"),
            line(2, row="r__x__one", what="words", words="the  earring\nlooks odd", frame=17),
            line(3, row="t83", what="words", words="a list row"),
            line(4, row="r__x__one", who="mrx (test of the page)", chosen="fine"),
            line(5, row="r__x__gone", chosen="do not use"),
            line(6, row="r__x__one", chosen=""),
            line(7, row="layers_view", what="words", words="super helpful"),
        ]))
        out = folder / "verdicts.jsonl"
        r = run("take", "--answers", answers, "--items", items, "--items", removed, "--out", out, "--dry-run")
        case("dry_run_writes_nothing", r.returncode == 0 and not out.exists() and "would write 4" in r.stdout, r.stdout + r.stderr)
        r = run("take", "--answers", answers, "--items", items, "--items", removed, "--out", out)
        got = [json.loads(t) for t in out.read_text().splitlines()]
        one = [v for v in got if v["row"] == "r__x__one"]
        case("a_choice_and_a_words_line_are_both_kept", len(one) == 2 and one[0]["chosen"] == "fine" and one[0]["render"] == "Video/x/one/one_00001.mp4"
             and one[0]["first"] == 10 and one[1]["frame"] == 17 and one[1]["source"] == "answers.jsonl:2", json.dumps(one))
        case("nothing_is_reworded", one[1]["words"] == "the  earring\nlooks odd", json.dumps(one[1]))
        case("a_line_on_a_list_row_is_left_to_its_tool", not any(v["row"] == "t83" for v in got) and "theirs to other tools) 1;" in r.stdout, r.stdout)
        case("another_persons_line_is_left_out", "not the owner's 1" in r.stdout and len(got) == 4, r.stdout)
        case("an_item_taken_off_the_page_is_still_known", any(v["row"] == "r__x__gone" and v["render"] == "Video/x/gone/gone_00001.mp4" for v in got), json.dumps(got))
        case("an_empty_submit_is_not_a_verdict", not any(v["source"] == "answers.jsonl:6" for v in got), json.dumps(got))
        case("a_view_that_is_not_a_render_is_kept_as_such", any(v["row"] == "layers_view" and v["render"] is None and v["words"] == "super helpful" for v in got), json.dumps(got))
        r2 = run("take", "--answers", answers, "--items", items, "--items", removed, "--out", out)
        case("a_second_run_writes_nothing", "written 0 line(s); already in the record 4" in r2.stdout and len(out.read_text().splitlines()) == 4, r2.stdout)
        r3 = run("show", "--verdicts", out, "--render", "one")
        case("show_prints_a_renders_lines_in_order", r3.returncode == 0 and "Video/x/one/one_00001.mp4" in r3.stdout
             and r3.stdout.index("fine") < r3.stdout.index("the  earring / looks odd") and "gone" not in r3.stdout, r3.stdout + r3.stderr)

    print(f"{'FAIL' if fails else 'ok'}: {fails} failing case(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
