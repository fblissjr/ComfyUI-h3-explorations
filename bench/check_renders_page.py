#!/usr/bin/env python3
"""`bench/renders_page.py` lays the renders inventory beside the owner's verdict lines; each case is a way a row could be out of order, carry another render's lines, drop a line or link to the wrong place.

Why it exists: the owner asked for one page on the share of every render in frame order with their own verdicts
under it; its only work is matching lines to renders by file and writing links that resolve from the share, and
each of those can be off by one row or one folder.

  rows_in_frame_order                   one row a render, sorted by first frame, not by the inventory's order
  lines_land_on_their_render            a choice and a words line on a render are under that render only, byte for byte
  a_render_with_no_line_says_so         "none yet", not another render's lines
  a_line_on_no_render_is_listed         a verdict whose render is in no inventory row is in the last section with its row id and the item's title
  links_resolve_from_the_share          the render link and the stacked clip link are relative to the page's folder under the root
  no_clip_is_said                       a render with no stacked clip of its name says so instead of linking to nothing
  out_outside_root_is_refused           by path

    <python> bench/check_renders_page.py

No model, no card; the standard library only.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "renders_page.py"

sys.path.insert(0, str(HERE))
from renders_page import relative  # noqa: E402


def fixture(root: Path) -> dict:
    (root / "Video/s1/late").mkdir(parents=True)
    (root / "Video/s1/early").mkdir(parents=True)
    (root / "Video/v/review_clips").mkdir(parents=True)
    (root / "Video/v/review_clips/0100_0159__s1__early.mp4").write_bytes(b"x")
    inv = {"video": "v", "read_from": "graphs", "made_at": "2026-10-12T00:00:00Z", "made_by": "check", "renders": [
        {"file": "Video/s1/late/late_00001.mp4", "session": "s1", "name": "late", "written": "2026-10-12T01:00:00Z", "first": 300, "frames": 10, "last": 309,
         "as_the_graph_says": {"replace": "whole subject"}, "motion_wired": False, "stills": ["who_b.jpg"], "in_the_joined_clip": False},
        {"file": "Video/s1/early/early_00001.mp4", "session": "s1", "name": "early", "written": "2026-10-12T02:00:00Z", "first": 100, "frames": 60, "last": 159,
         "as_the_graph_says": {"replace": "the wired parts", "part_phrases": "hair, head"}, "motion_wired": True, "stills": ["who_a.jpg"], "in_the_joined_clip": True}]}
    (root / "renders.json").write_text(json.dumps(inv))
    lines = [
        {"at": "2026-10-11T10:00:00Z", "what": "choice", "chosen": "kept", "render": "Video/s1/early/early_00001.mp4", "row": "r__s1__early", "first": 100, "frames": 60},
        {"at": "2026-10-11T10:00:01Z", "what": "words", "words": "WORDS ON EARLY  <b>", "render": "Video/s1/early/early_00001.mp4", "row": "r__s1__early", "first": 100, "frames": 60},
        {"at": "2026-10-11T10:00:02Z", "what": "words", "words": "WORDS ON A GATE CLIP", "render": None, "row": "gate_x", "first": 14, "frames": 141, "title": "t"}]
    (root / "verdicts.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (root / "items.json").write_text(json.dumps({"items": [{"id": "gate_x", "title": "THE GATE ITEM", "path": "data/v/review/media/g.mp4"}]}))
    return {"root": root, "renders": root / "renders.json", "verdicts": root / "verdicts.jsonl", "items": root / "items.json", "out": root / "Video/v/renders"}


def run(p, out=None, *extra):
    cmd = [sys.executable, str(TOOL), "--renders", str(p["renders"]), "--verdicts", str(p["verdicts"]), "--items", str(p["items"]),
           "--clips", "Video/v/review_clips", "--root", str(p["root"]), "--out", str(out or p["out"]), *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    fails = 0

    def case(name, ok, why=""):
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {why}"))
        fails += not ok

    with tempfile.TemporaryDirectory() as d:
        p = fixture(Path(d))
        r = run(p)
        assert r.returncode == 0, r.stdout + r.stderr
        page = (p["out"] / "index.html").read_text()
        main_table = page.split("<h2>")[0]
        rows = main_table.split("<tr>")[2:]     # [0] before the table, [1] the header row
        case("rows_in_frame_order", len(rows) == 2 and "100-159" in rows[0] and "300-309" in rows[1], str([x[:40] for x in rows]))
        case("lines_land_on_their_render", "WORDS ON EARLY  &lt;b&gt;" in rows[0] and ">kept<" in rows[0] and "WORDS ON EARLY" not in rows[1], "")
        case("a_render_with_no_line_says_so", "none yet" in rows[1] and "kept" not in rows[1], "")
        tail = page.split("<h2>")[1] if "<h2>" in page else ""
        case("a_line_on_no_render_is_listed", "gate_x" in tail and "THE GATE ITEM" in tail and "WORDS ON A GATE CLIP" in tail and "WORDS ON A GATE CLIP" not in main_table, tail[:200])
        case("links_resolve_from_the_share", 'href="../../s1/early/early_00001.mp4"' in rows[0] and 'href="../review_clips/0100_0159__s1__early.mp4"' in rows[0]
             and relative("Video/s1/x.mp4", "Video/v/renders") == "../../s1/x.mp4" and relative("Video/v/review_clips/c.mp4", "Video/v/renders") == "../review_clips/c.mp4", rows[0][-400:])
        case("no_clip_is_said", "no stacked clip on the share" in rows[1] and "review_clips" not in rows[1], "")
        with tempfile.TemporaryDirectory() as elsewhere:     # a second temporary folder, so it is not under the fixture's root
            r = run(p, Path(elsewhere) / "renders")
            case("out_outside_root_is_refused", r.returncode != 0 and "not under" in r.stdout + r.stderr, r.stdout + r.stderr)

    print(f"{'FAIL' if fails else 'ok'}: {fails} failing case(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
