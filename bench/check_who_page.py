#!/usr/bin/env python3
"""`bench/who_page.py` lays the shots record beside each tracker's numbered picture; each case is a way a row could show the wrong band, the wrong words or nothing.

Why it exists: the owner asked for one simple page of who is in each shot with the masks and identifiers on the
input; its only work is cutting the right band of a table's picture for the right shot and putting the record's
own words beside it, and each of those can be off by one.

  a_row_a_shot                          the page has one row per shot of the record, with its frames and each subject's line
  the_band_cut_is_the_shots_own         band k of the picture, by colour, for the table shot that overlaps the record's shot
  a_band_from_another_preview_is_not_cut   a preview whose job does not span the shot gives no tile
  empty_bands_collapse_to_one           bands with nobody on them are said in words, never shown as black pictures
  the_owners_words_are_on_their_frames  a words line whose source frame is inside the shot is on that row and no other
  a_missing_table_is_refused            by path
  a_long_shot_asks_at_its_quarters      with --source, a shot longer than --long-shot-frames shows its first- and last-quarter
                                        frames plain with the two frame buttons, beside the numbered picture; a shorter shot does not
  quarter_frames_are_the_shots_own      the pure rule: nothing under the line, the quarters of the shot's own length over it
  a_span_shows_only_its_shots           with --span, a shot with no frame inside it has no row, and the header names the span
  the_address_is_the_outs               the page says it is answered from /<the --out folder's name>/index.html, not from /who/
  a_page_is_not_written_over_another    an --out holding a page of another span, or one that does not say its span, is refused
                                        with its bytes as they were; the same span is built again in place
  a_sessions_reading_is_a_guess_in_a_closed_fold   with --notes, a preview's line is under that preview's tiles and no other, in a
                                        fold that is not open, headed as a session's guess; it holds no button and presses
                                        none; a line naming no preview given is printed; with no notes there is no fold
  a_guess_says_when_its_number_is_of_another_picture   a note's person number is of its subject's own tracker's picture; where
                                        that tracker showed another frame than the picture on the page, the fold says so in capitals
  a_tile_asks_only_its_jobs_subjects    a numbered picture has buttons for the subjects its preview's job names and for no other
                                        subject of the record, who is named under it in words
  a_shot_is_shown_cut_by_cut            with --every-cut-of, a record shot the tracker cut into five shows the cuts as long as the
                                        shortest load and no shorter one, each with its own buttons on its own frame; a long cut
                                        with nobody outlined is asked about plain; the shorter cuts are counted and named; the
                                        same shot without the flag has one picture; a span that is no shot of the record is refused
  the_boxes_are_each_persons_own        the picture of boxes is the table's own size, each person's box is drawn in that person's
                                        colour where the table puts it, the number buttons carry the same colours, and the page
                                        says a box is a box

    <python> bench/check_who_page.py

No model, no card; PIL, and ffmpeg for the one case that cuts a frame from a source (not graded without it).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from who_page import BOXES_SAY, colour_of, quarter_frames  # noqa: E402

HERE = Path(__file__).resolve().parent
TOOL = HERE / "who_page.py"
COLOURS = [(200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0)]


def table(shots):
    return {"version": 2, "size": [64, 48], "shots": shots}


def picture(path: Path, n: int):
    im = Image.new("RGB", (64, 48 * n))
    for k in range(n):
        im.paste(COLOURS[k], (0, 48 * k, 64, 48 * (k + 1)))
    im.save(path)


def fixture(folder: Path) -> dict:
    tables = folder / "t"
    tables.mkdir()
    # preview pv: first 100, two subjects; its table: four shots, the first two with nobody (a lead-in), then two with people
    shots = [{"shot": 1, "first_frame": 0, "last_frame": 0, "shown_frame": 0, "people": []},
             {"shot": 2, "first_frame": 1, "last_frame": 1, "shown_frame": 1, "people": []},
             {"shot": 3, "first_frame": 2, "last_frame": 29, "shown_frame": 10, "people": [{"person": 1, "box": [1, 1, 5, 5]}]},
             {"shot": 4, "first_frame": 30, "last_frame": 59, "shown_frame": 40, "people": [{"person": 1, "box": [1, 1, 5, 5]}, {"person": 2, "box": [9, 9, 5, 5]}]}]
    for label in ("a", "b"):
        (tables / f"pv_{label}_shots_00001_shots.json").write_text(json.dumps(table(shots)))
        picture(tables / f"pv_{label}_shots_00001_shots.png", 4)
    job = folder / "pv.json"
    job.write_text(json.dumps({"job": {"name": "pv", "first": 100, "frames": 60, "subjects": [{"label": "a"}, {"label": "b"}]}}))
    # a second preview that spans other frames entirely
    (tables / "far_a_shots_00001_shots.json").write_text(json.dumps(table(shots[2:3])))
    picture(tables / "far_a_shots_00001_shots.png", 1)
    far = folder / "far.json"
    far.write_text(json.dumps({"job": {"name": "far", "first": 900, "frames": 30, "subjects": [{"label": "a"}]}}))
    rec = folder / "shots.json"
    rec.write_text(json.dumps({"schema": "h3_shots/1", "video": "t.mp4", "span": [100, 159], "subjects": ["a", "b"], "shots": [
        {"first": 100, "last": 101, "who": {"a": {"present": "no", "how": "lead-in", "by": "x"}, "b": {"present": "no", "how": "lead-in", "by": "x"}}},
        {"first": 102, "last": 129, "who": {"a": {"present": "yes", "how": "tile 110", "by": "owner"}, "b": {"present": "no", "how": "", "by": "x"}}},
        {"first": 130, "last": 159, "who": {"a": {"present": "unsure", "how": "two say", "by": "x"}, "b": {"present": "yes", "how": "tile 140", "by": "owner"}}}]}))
    ans = folder / "answers.jsonl"
    ans.write_text("".join(json.dumps(d) + "\n" for d in [
        {"who": "owner", "row": "who", "what": "words", "source_frame": 110, "at": "2026-10-12T10:00:00Z", "words": "WORDS ON 110"},
        {"who": "owner", "row": "who", "what": "words", "source_frame": 140, "at": "2026-10-12T10:00:01Z", "words": "WORDS ON 140"},
        {"who": "mrx", "row": "who", "what": "words", "source_frame": 110, "at": "2026-10-12T10:00:02Z", "words": "NOT THE OWNER"}]))
    return {"shots": rec, "pv": f"{job}={tables}", "far": f"{far}={tables}", "answers": ans, "out": folder / "out"}


def run(p, *extra):
    cmd = [sys.executable, str(TOOL), "--shots", str(p["shots"]), "--answers", str(p["answers"]), "--out", str(p["out"]), *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    fails = 0

    def case(name, ok, why=""):
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {why}"))
        fails += not ok

    with tempfile.TemporaryDirectory() as d:
        p = fixture(Path(d))
        r = run(p, "--preview", p["pv"], "--preview", p["far"])
        assert r.returncode == 0, r.stdout + r.stderr
        page = (p["out"] / "index.html").read_text()
        rows = page.count("<tr>") - 1
        case("a_row_a_shot", rows == 3 and "100-101" in page and "130-159" in page and "<b>a</b>: UNSURE" in page, str(rows))
        tiles = sorted(x.name for x in (p["out"] / "tiles").glob("*.jpg"))
        band3 = Image.open(p["out"] / "tiles" / "0102_pv_a_shot3.jpg").getpixel((32, 24))
        band4 = Image.open(p["out"] / "tiles" / "0130_pv_b_shot4.jpg").getpixel((32, 24))
        near = lambda c, want: all(abs(int(x) - y) < 20 for x, y in zip(c, want))
        case("the_band_cut_is_the_shots_own", near(band3, COLOURS[2]) and near(band4, COLOURS[3]), f"{band3} {band4}")
        case("a_band_from_another_preview_is_not_cut", not any("far" in t for t in tiles), str(tiles))
        case("empty_bands_collapse_to_one", not any(t.startswith("0100_") for t in tiles) and "found nobody" in page.split("<tr>")[2], str(tiles))
        row2 = page.split("<tr>")[3]   # [1] is the header row, [2] the lead-in
        row3 = page.split("<tr>")[4]
        case("the_owners_words_are_on_their_frames", "WORDS ON 110" in row2 and "WORDS ON 140" not in row2 and "WORDS ON 140" in row3
             and "NOT THE OWNER" not in page, "")
        r = run(p, "--preview", f"{Path(d) / 'pv.json'}={Path(d) / 'nowhere'}")
        case("a_missing_table_is_refused", r.returncode != 0 and "no table" in r.stdout + r.stderr, r.stdout + r.stderr)
        case("the_address_is_the_outs", ":8765/out/index.html" in page and "/who/index.html" not in page
             and json.loads((p["out"] / "page.json").read_text())["span"] == [100, 159], page[:1500])
        own = Path(d) / "part"
        r = run({**p, "out": own}, "--preview", p["pv"], "--span", "125-159")
        part = (own / "index.html").read_text() if r.returncode == 0 else r.stdout + r.stderr
        case("a_span_shows_only_its_shots", r.returncode == 0 and part.count("<tr>") - 1 == 2 and "100-101" not in part and "102-129" in part
             and "the shots of frames 125-159 (the record holds 100-159)" in part and ":8765/part/index.html" in part, part[:1500])
        before = (p["out"] / "index.html").read_bytes()
        r = run(p, "--preview", p["pv"], "--span", "125-159")
        bare = Path(d) / "bare"
        bare.mkdir()
        (bare / "index.html").write_text("a page made before a page said its span")
        r2 = run({**p, "out": bare}, "--preview", p["pv"])
        r3 = run({**p, "out": own}, "--preview", p["pv"], "--span", "125-159")
        case("a_page_is_not_written_over_another", r.returncode != 0 and "holds the page of frames 100-159 and this one is of 125-159" in r.stdout + r.stderr
             and (p["out"] / "index.html").read_bytes() == before and r2.returncode != 0 and "does not say its span" in r2.stdout + r2.stderr
             and (bare / "index.html").read_text() == "a page made before a page said its span" and r3.returncode == 0,
             r.stdout + r.stderr + r2.stdout + r2.stderr + r3.stdout + r3.stderr)
        notes = Path(d) / "run.jsonl"
        notes.write_text("".join(json.dumps(x) + "\n" for x in [
            # the first line is of another preview, and of this preview's name on another first frame: neither is this tile's
            {"first": 700, "job_file": "inputs/elsewhere.json", "subjects": [{"label": "a", "picked_number": 1, "why": "NOT THIS PREVIEW'S"}]},
            {"first": 101, "job_file": "inputs/pv.json", "subjects": [{"label": "a", "picked_number": 1, "why": "NOT THIS PREVIEW'S"}]},
            {"first": 100, "job_file": "inputs/pv.json", "recorded_at": "2026-10-13T15:00:00Z", "subjects": [
                {"label": "a", "picked_number": 2, "picked_looks_like": "A TALL ONE <b>", "right_person": "no", "why": "THE SMALL ONE IS a"}]}]))
        noted = Path(d) / "noted"
        r = run({**p, "out": noted}, "--preview", p["pv"], "--preview", p["far"], "--notes", str(notes))
        text = (noted / "index.html").read_text() if r.returncode == 0 else ""
        fold = text.split("<details", 1)[1].split("</details>", 1)[0] if "<details" in text else ""
        case("a_sessions_reading_is_a_guess_in_a_closed_fold", r.returncode == 0 and text.count('<details class="guess">') == 2
             and "<details open" not in text and "a session's guess, NOT the owner's" in fold and "run.jsonl line 3, recorded 2026-10-13T15:00:00Z" in fold and "NOT THIS PREVIEW" not in text
             and "picked person 2 (A TALL ONE &lt;b&gt;)" in fold and "<b>no</b>" in fold and "THE SMALL ONE IS a" in fold
             and "<button" not in fold and 'class="on"' not in text and "data-chosen" not in fold
             and "elsewhere (first frame 700), run.jsonl line 1" in r.stdout and "pv (first frame 101), run.jsonl line 2" in r.stdout and "<details" not in page, r.stdout + r.stderr + fold)
        # a note's number is of its own tracker's picture: where that is another frame than the picture shown, the fold says so
        (Path(d) / "t" / "pv_b_shots_00001_shots.json").write_text(json.dumps(table([
            {"shot": 1, "first_frame": 0, "last_frame": 0, "shown_frame": 0, "people": []},
            {"shot": 2, "first_frame": 1, "last_frame": 1, "shown_frame": 1, "people": []},
            {"shot": 3, "first_frame": 2, "last_frame": 29, "shown_frame": 20, "people": [{"person": 1, "box": [1, 1, 5, 5]}]},
            {"shot": 4, "first_frame": 30, "last_frame": 59, "shown_frame": 40, "people": [{"person": 1, "box": [1, 1, 5, 5]}, {"person": 2, "box": [9, 9, 5, 5]}]}])))
        notes.write_text(json.dumps({"first": 100, "job_file": "inputs/pv.json", "subjects": [
            {"label": "a", "picked_number": 1, "why": "ON a's OWN"}, {"label": "b", "picked_number": 1, "why": "ON b's OWN"}]}) + "\n")
        other = Path(d) / "other_frame"
        r = run({**p, "out": other}, "--preview", p["pv"], "--notes", str(notes))
        text = (other / "index.html").read_text() if r.returncode == 0 else ""
        row2, row3 = text.split("<tr>")[3], text.split("<tr>")[4]
        case("a_guess_says_when_its_number_is_of_another_picture", r.returncode == 0 and 'data-frame="110"' in row2
             and row2.count("THAT NUMBER IS NOT OF THE PICTURE ABOVE") == 1 and "it is of b's own tracker's picture (frame 120), and the picture above is frame 110" in row2
             and "THAT NUMBER IS NOT OF THE PICTURE ABOVE" not in row3, r.stdout + r.stderr + row2[-1500:])
        # the same tables under a job that names `a` alone: `b` is a subject of the record and not of the job
        solo = Path(d) / "solo.json"
        solo.write_text(json.dumps({"job": {"name": "pv", "first": 100, "frames": 60, "subjects": [{"label": "a"}]}}))
        r = run(p, "--preview", f"{solo}={Path(d) / 't'}")
        assert r.returncode == 0, r.stdout + r.stderr
        row3 = (p["out"] / "index.html").read_text().split("<tr>")[4]
        case("a_tile_asks_only_its_jobs_subjects", row3.count('data-sub="a"') == 4 and 'data-sub="b"' not in row3
             and "not asked here: b." in row3 and page.split("<tr>")[4].count('data-sub="b"') == 4, row3)
        case("quarter_frames_are_the_shots_own",
             quarter_frames({"first": 102, "last": 129}, 20) == [109, 123] and quarter_frames({"first": 102, "last": 129}, 28) == []
             and quarter_frames({"first": 604, "last": 1009}) == [705, 908], str(quarter_frames({"first": 102, "last": 129}, 20)))
        if shutil.which("ffmpeg"):
            src = Path(d) / "src.mp4"      # 160 frames of flat colour, so every frame the page asks for exists
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x48:r=24:d=7", "-frames:v", "160",
                            "-pix_fmt", "yuv420p", str(src)], check=True)
            r = run(p, "--preview", p["pv"], "--source", str(src), "--long-shot-frames", "20")
            assert r.returncode == 0, r.stdout + r.stderr
            page = (p["out"] / "index.html").read_text()
            tiles = sorted(x.name for x in (p["out"] / "tiles").glob("*_plain.jpg"))
            row2, row3 = page.split("<tr>")[3], page.split("<tr>")[4]
            case("a_long_shot_asks_at_its_quarters",
                 {"0109_plain.jpg", "0123_plain.jpg", "0137_plain.jpg", "0152_plain.jpg"} <= set(tiles)
                 and row2.count('data-table="plain"') == 2 and 'data-frame="109"' in row2 and 'data-frame="123"' in row2
                 and 'data-frame="110"' in row2 and "first quarter" in row2 and "last quarter" in row3, str(tiles))
            r = run(p, "--preview", p["pv"], "--source", str(src))
            page = (p["out"] / "index.html").read_text()
            case("a_short_shot_asks_once", page.split("<tr>")[3].count('data-table="plain"') == 0, "")
            # one shot of the record that the tracker cut into five: a one-frame cut, two long ones with people, a long one with nobody, a two-frame tail
            import masked_job
            least = masked_job.shortest_load()[1]
            cuts = [{"shot": 1, "first_frame": 0, "last_frame": 0, "shown_frame": 0, "people": [{"person": 1, "box": [1, 1, 5, 5]}]},
                    {"shot": 2, "first_frame": 1, "last_frame": least, "shown_frame": 3,
                     "people": [{"person": 1, "box": [20, 20, 100, 150]}, {"person": 2, "box": [170, 50, 120, 150]}]},
                    {"shot": 3, "first_frame": least + 1, "last_frame": 3 * least, "shown_frame": least + 2, "people": [{"person": 1, "box": [10, 10, 30, 30]}]},
                    {"shot": 4, "first_frame": 3 * least + 1, "last_frame": 4 * least + 3, "shown_frame": 3 * least + 5, "people": []},
                    {"shot": 5, "first_frame": 4 * least + 4, "last_frame": 4 * least + 5, "shown_frame": 4 * least + 4, "people": [{"person": 1, "box": [1, 1, 5, 5]}]}]
            total = 4 * least + 6
            # a table large enough that a number's plate at a box's middle is clear of the box's own edges
            (Path(d) / "t" / "cut_a_shots_00001_shots.json").write_text(json.dumps({**table(cuts), "size": [320, 240]}))
            im = Image.new("RGB", (64, 48 * 5))
            for k in range(5):
                im.paste(COLOURS[k % 4], (0, 48 * k, 64, 48 * (k + 1)))
            im.save(Path(d) / "t" / "cut_a_shots_00001_shots.png")
            (Path(d) / "cut.json").write_text(json.dumps({"job": {"name": "cut", "first": 100, "frames": total, "subjects": [{"label": "a"}]}}))
            one = Path(d) / "one_shot.json"
            one.write_text(json.dumps({"schema": "h3_shots/1", "video": "t.mp4", "span": [100, 99 + total], "subjects": ["a", "b"], "shots": [
                {"first": 100, "last": 99 + total, "who": {"a": {"present": "yes", "how": "", "by": "x"}, "b": {"present": "no", "how": "", "by": "x"}}}]}))
            long_src = Path(d) / "long.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=128x90:r=24:d=60", "-frames:v", str(101 + total),
                            "-pix_fmt", "yuv420p", str(long_src)], check=True)       # 128x90 is not the table's shape: the frame is cut to it, as the loader does
            q = {**p, "shots": one, "out": Path(d) / "cuts"}
            r = run(q, "--preview", f"{Path(d) / 'cut.json'}={Path(d) / 't'}", "--source", str(long_src), "--every-cut-of", f"100-{99 + total}", "--long-shot-frames", "100000")
            text = (q["out"] / "index.html").read_text() if r.returncode == 0 else ""
            numbered = [t for t in text.split('<div class="tile"')[1:] if 'data-table="plain"' not in t]
            plain = [t for t in text.split('<div class="tile"')[1:] if 'data-table="plain"' in t]
            r1 = run({**q, "out": Path(d) / "once"}, "--preview", f"{Path(d) / 'cut.json'}={Path(d) / 't'}", "--source", str(long_src), "--long-shot-frames", "100000")
            once = (Path(d) / "once" / "index.html").read_text() if r1.returncode == 0 else ""
            r2 = run({**q, "out": Path(d) / "no"}, "--preview", f"{Path(d) / 'cut.json'}={Path(d) / 't'}", "--every-cut-of", "100-120")
            case("a_shot_is_shown_cut_by_cut", r.returncode == 0 and len(numbered) == 2 and 'data-frame="103"' in numbered[0] and 'data-shot="2"' in numbered[0]
                 and f'data-frame="{102 + least}"' in numbered[1] and 'data-shot="3"' in numbered[1] and numbered[0].count('data-sub="a"') == 4
                 and numbered[1].count('data-sub="a"') == 3 and len(plain) == 1 and f'data-frame="{105 + 3 * least}"' in plain[0]
                 and f"the tracker outlined nobody on its cut {101 + 3 * least}-{103 + 4 * least}" in plain[0]
                 and f"cut it into 5; the 3 of {least} frames or more" in text and f"holding {total - 3} of the shot's {total} frames" in text
                 and f"2 shorter cut(s) are not shown: 100-100, {104 + 4 * least}-{105 + 4 * least}." in text and text.count("This shot is shown cut by cut") == 1
                 and r1.returncode == 0 and once.count('<div class="tile"') == 1 and "_boxes" not in once and "cut by cut" not in once
                 and r2.returncode != 0 and "is not a shot of the record" in r2.stdout + r2.stderr and not (Path(d) / "no" / "index.html").exists(),
                 r.stdout + r.stderr + r2.stdout + r2.stderr + f" numbered {len(numbered)} plain {len(plain)} once {once.count('tile')}")
            boxes = q["out"] / "tiles" / "0103_cut_a_shot2_boxes.jpg"
            fitted = q["out"] / "tiles" / "0103_as_tracked_320x240.jpg"
            if boxes.is_file() and fitted.is_file():
                im = Image.open(boxes).convert("RGB")
                left, right, off = im.getpixel((20, 40)), im.getpixel((289, 180)), im.getpixel((150, 30))     # person 1's left edge, person 2's right edge, between them
                plate1, plate2 = im.getpixel((70, 95)), im.getpixel((230, 125))                              # each box's middle: a dark plate or the number's ink
                yellow, cyan = colour_of(1), colour_of(2)
                ok = (im.size == (320, 240) == Image.open(fitted).size and left[0] - left[2] > 100 and right[2] - right[0] > 100 and abs(off[0] - off[2]) < 25
                      and plate1 != off and plate2 != off and Image.open(fitted).getpixel((70, 95))[0] == Image.open(fitted).getpixel((150, 30))[0]
                      and yellow[0] > yellow[2] and cyan[2] > cyan[0]
                      and 'data-chosen="1" style="color:rgb(%d,%d,%d);border-color:rgb(%d,%d,%d)"' % (yellow * 2) in numbered[0]
                      and 'data-chosen="2" style="color:rgb(%d,%d,%d);border-color:rgb(%d,%d,%d)"' % (cyan * 2) in numbered[0]
                      and BOXES_SAY in numbered[0] and "0103_cut_a_shot2_boxes.jpg" in numbered[0] and "0103_as_tracked_320x240.jpg" in numbered[0]
                      and 'style="color' not in once)
                case("the_boxes_are_each_persons_own", ok, f"{im.size} left {left} right {right} off {off}")
            else:
                case("the_boxes_are_each_persons_own", False, f"no picture of boxes: {sorted(x.name for x in (q['out'] / 'tiles').glob('*'))}")
        else:
            print("not graded: a_long_shot_asks_at_its_quarters, a_shot_is_shown_cut_by_cut, the_boxes_are_each_persons_own (no ffmpeg on the path)")

    print(f"{'FAIL' if fails else 'ok'}: {fails} failing case(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
