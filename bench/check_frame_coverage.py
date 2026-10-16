#!/usr/bin/env python3
"""`bench/frame_coverage.py` on a made table and two made captures: each case is a way a frame where a tracked person
shows as the original could be missed, given the wrong cause, or called "not tracked" when nothing could tell.

No render and no share: the tool reads the table's JSON and the capture records (`manifest.json`, the subject's
`masks__<by>.npz`, `part_grades.json`, the run's `region.npz`), so the fixture is those files, written here.

The made shot is 10-29, subject `a`. Piece `ra` (capture one, 10-29) lays 12-19; piece `rb` (capture two, 20-24)
lays 20-24. Inside `ra` the node carried nothing on 14 (`drop=`, its saved part empty too), 15 (the floor), 16 (the grade off its classes),
17 (nothing saved) and 18 (her track empty there too); 13 is held. Outside the pieces she is tracked on 10-11 and
25-26, her track is empty on 27, capture one covers 28-29 with no track of her there, and no tied capture covers
30-31, where an untied capture of the same clip holds her track (named, not read). The shot is `unsure`.

1. **The original while tracked**: 10-11 and 25-26 with no piece, and 14, 15, 16, 17 each with its cause; 18 is not
   in it (her track is empty) and 13 is not (held, carried).
2. **Seams**: 19/20, `ra` meeting `rb`, each with its render day.
3. **Emptied inside a piece**: the four with their causes, and nothing else.
4. **Not known is not "not tracked"**: 28-29 and 30-31 come back as "tracking not known", each with why, and 30-31
   name the untied capture; 18 and 27 (her track 0 px) are their own list; the unsure shot is read and said.
5. **A row whose run= does not resolve** is a hard failure and the tool exits 1, and its frames are not known, not
   "no piece laid".
6. **A capture tied to a row of another shot** that reaches this shot's frames is read for her track there, and a
   frame whose piece's mask is not known while her track is known is `carried_not_known`, not "tracking not known"
   (the outside reviewer's D3, bugs A and B).
7. **A look and a job's left_out** (the lead's ruling of 2026-10-16, as built): frames the shots record says are not
   seen are "left out by a look", with who and why, and out of list 1; a later `seen` takes them back; a laid
   piece's job left_out is said beside its frames and leaves them in list 1, since it may say "left for a later load".
8. **The ledger reads the same rule**: `render_ledger.left_by_the_load` given the run gives 14-18 by cause; with no
   run it falls back to the manifest's `drop=` list and says so.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import frame_coverage as fc  # noqa: E402
import render_ledger as led  # noqa: E402

TOOL = Path(__file__).resolve().parent / "frame_coverage.py"
H, W = 8, 16


def packed(on: list[bool], n: int) -> np.ndarray:
    m = np.zeros((n, H, W), bool)
    for k, v in enumerate(on):
        if v:
            m[k, 2:6, 3:11] = True
    return np.packbits(m, axis=-1)


def capture(root: Path, name: str, first: int, n: int, track: list[bool], covered: list[bool], parts: list[bool],
            carried: list[bool], run: str | None, sighting: dict, grades: list[dict]) -> Path:
    d = root / name
    (d / "subjects" / "a").mkdir(parents=True)
    np.savez(d / "subjects" / "a" / "masks__p.npz", track=packed(track, n), covered=np.array(covered), parts=packed(parts, n))
    (d / "subjects" / "a" / "part_grades.json").write_text(json.dumps({"frames": grades}))
    runs = []
    if run:
        (d / "runs" / run).mkdir(parents=True)
        np.savez(d / "runs" / run / "region.npz", carried=packed(carried, n), read=np.ones(n, bool), region=np.zeros((n, 1, 1), bool))
        runs.append({"name": run, "subject": "a", "left_as_the_source_with_no_mask": [], "left_as_the_source_across_a_cut": []})
    (d / "manifest.json").write_text(json.dumps({"clip": "c.mov", "first_frame": first, "frames": n, "runs": runs,
                                                 "subjects": [{"label": "a", "sightings": [{"by": "p", **sighting}]}]}))
    return d


def made(root: Path) -> Path:
    f = range(10, 30)
    one = capture(root, "2026-01-01_ra", 10, 20,
                  track=[x not in (18, 27) for x in f], covered=[x < 28 for x in f], parts=[x not in (14, 17) for x in f],
                  carried=[x not in (14, 15, 16, 17, 18) for x in f], run="ra",
                  sighting={"parts_emptied_on_source_frames": [[14, 14]], "parts_emptied_by_grade_on_source_frames": [[15, 16]],
                            "parts_held_on_source_frames": [[13, 13]]},
                  grades=[{"source_frame": 15, "did": "emptied", "why": "too small to be drawn as the part"},
                          {"source_frame": 16, "did": "emptied", "why": "not on its own classes"}])
    two = capture(root, "2026-01-02_rb", 20, 5, track=[True] * 5, covered=[True] * 5, parts=[True] * 5, carried=[True] * 5, run="rb",
                  sighting={}, grades=[])
    capture(root, "2026-01-03_untied", 30, 2, track=[True] * 2, covered=[True] * 2, parts=[True] * 2, carried=[True] * 2, run=None,
            sighting={}, grades=[])
    row = lambda name, a, b, cap: {"first": a, "last": b, "name": name, "file": f"Video/v/{name}_00001.mp4", "subject": "a",  # noqa: E731
                                   "run": f"{cap}:{name}"}
    table = {"rows": [row("ra", 12, 19, one), row("rb", 20, 24, two)],
             "reading": [{"first": 10, "last": 31, "subject": "a", "present": "unsure",
                          "stands": [{"render": "ra", "written": "2026-01-01T18:00:00Z"}, {"render": "rb", "written": "2026-01-02T18:00:00Z"}]}]}
    (root / "t.json").write_text(json.dumps(table))
    return root / "t.json"


def runs_of(items) -> list:
    return [(tuple(x["frames"]), x["piece"], x["state"]) for x in items]


def original_while_tracked() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out = fc.coverage(json.loads(made(Path(tmp)).read_text()), data=Path(tmp))
        got = runs_of(out["original_while_tracked"])
        want = [((10, 11), None, "original: no piece laid"),
                ((14, 14), "ra", "original: emptied inside the piece (drop=+nothing saved)"),
                ((15, 15), "ra", "original: emptied inside the piece (floor)"),
                ((16, 16), "ra", "original: emptied inside the piece (grade)"),
                ((17, 17), "ra", "original: emptied inside the piece (nothing saved)"),
                ((25, 26), None, "original: no piece laid")]
        assert got == want, got
        st = dict(zip(out["shots"][0]["frames"], out["shots"][0]["state"]))
        assert st[13] == "carried (held from a neighbouring frame)" and st[18].startswith("original: emptied") and st[27] == "original: no piece laid", st
        assert out["original_while_tracked"][0]["track_px"] == {"least": 32, "median": 32, "most": 32}, out["original_while_tracked"][0]
    return "no-piece frames and emptied frames each with its cause; an empty track and a held frame left out"


def seams() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out = fc.coverage(json.loads(made(Path(tmp)).read_text()), data=Path(tmp))
        (s,) = out["seams"]
        assert s["at"] == [19, 20] and (s["before"]["piece"], s["before"]["day"], s["after"]["piece"], s["after"]["day"]) == ("ra", "01/01", "rb", "01/02"), s
        kinds = [(r["first"], r["last"], r["kind"], r["pieces"], r["cause"]) for r in out["runs"]]
        assert kinds == [(10, 11, "original_while_tracked", [], "no piece laid"), (14, 14, "emptied_inside", ["ra"], "drop=+nothing saved"),
                         (15, 15, "emptied_inside", ["ra"], "floor"), (16, 16, "emptied_inside", ["ra"], "grade"),
                         (17, 17, "emptied_inside", ["ra"], "nothing saved"),
                         (18, 18, "original_her_track_empty", ["ra"], "original: emptied inside the piece (not in the record)"),
                         (19, 20, "seam", ["ra", "rb"], None),
                         (25, 26, "original_while_tracked", [], "no piece laid"),
                         (27, 27, "original_her_track_empty", [], "original: no piece laid"),
                         (28, 29, "original_tracking_not_known", [], "a tied capture covers this frame and holds no track of her on it: tracking not known"),
                         (30, 31, "original_tracking_not_known", [], "no tied capture covers this frame: tracking not known")], kinds
        assert out["runs"][6]["render_days"] == ["01/01", "01/02"] and out["runs"][1]["render_days"] == ["01/01"], out["runs"]
    return "19/20 with both render days; the page's one list of runs, by frame, kinds disjoint"


def emptied_inside() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out = fc.coverage(json.loads(made(Path(tmp)).read_text()), data=Path(tmp))
        assert [x["frames"] for x in out["emptied_inside_a_piece"]] == [[14, 14], [15, 15], [16, 16], [17, 17]], out["emptied_inside_a_piece"]
        out["counts"] = fc.counts(out)
        c = out["counts"]
        assert c["emptied_inside_a_piece"]["frames_by_cause"] == {"drop=+nothing saved": 1, "floor": 1, "grade": 1, "nothing saved": 1}, c
        assert c["original_and_her_track_empty_frames"] == 2, c       # 18 (inside ra) and 27
    return "the four by cause; 18 counted as her track empty"


def not_known() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out = fc.coverage(json.loads(made(Path(tmp)).read_text()), data=Path(tmp))
        u, v = out["original_tracking_not_known"]
        assert u["frames"] == [28, 29] and u["tracking"].endswith("holds no track of her on it: tracking not known") \
            and u["untied_captures_with_her_track"] == [], u
        assert v["frames"] == [30, 31] and v["tracking"] == "no tied capture covers this frame: tracking not known" \
            and v["untied_captures_with_her_track"] == ["2026-01-03_untied"], v
        i = out["shots"][0]["frames"].index(28)
        assert out["shots"][0]["track_px"][i] is None, out["shots"][0]["track_px"]
        assert [x["frames"] for x in out["original_her_track_empty"]] == [[18, 18], [27, 27]], out["original_her_track_empty"]
        assert out["shots"][0]["present"] == "unsure" and all(x["present"] == "unsure" for x in out["runs"] if x["kind"] != "seam"), out["runs"]
        assert not any("2026-01-03_untied" in x["track_from"] for x in out["original_while_tracked"]), "an untied capture was read"
    return "28-29 and 30-31 not known with why, the untied capture named and not read; her empty track its own list; unsure read and said"


def unresolved_is_red() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        t = made(Path(tmp))
        table = json.loads(t.read_text())
        table["rows"].append({"first": 25, "last": 26, "name": "gone", "file": "Video/v/gone.mp4", "subject": "a", "run": f"{Path(tmp) / 'nope'}:gone"})
        table["rows"].append({"first": 27, "last": 27, "name": "norun", "file": "Video/v/norun.mp4", "subject": "a",
                              "run": f"{Path(tmp) / '2026-01-01_ra'}:norun"})
        t.write_text(json.dumps(table))
        r = subprocess.run([sys.executable, str(TOOL), "--table", str(t)], capture_output=True, text=True)
        assert r.returncode == 1 and "FAIL row gone: run=" in r.stdout and "no run 'norun'" in r.stdout, r.stdout + r.stderr
        out = fc.coverage(json.loads(t.read_text()), data=Path(tmp))
        st = dict(zip(out["shots"][0]["frames"], out["shots"][0]["state"]))
        assert st[25] == "not known: the laid piece gone has a run= that does not resolve" and st[27].startswith("not known: the laid piece norun"), st
        # her track is known on 25-26 (32 px) and 27 (0 px): the piece's mask is what is not known there, not her tracking
        assert [(x["frames"], x["track_px"]["median"]) for x in out["carried_not_known"]] == [([25, 26], 32), ([27, 27], 0)], out["carried_not_known"]
        assert not any(x["frames"][0] <= 26 and x["frames"][1] >= 25 for x in out["original_tracking_not_known"]), out["original_tracking_not_known"]
        assert (Path(tmp) / "frame_coverage_t.json").is_file() and (Path(tmp) / "frame_coverage_t.md").read_text().count("## FAILED") == 1
    return "a folder that does not exist and a run its manifest lacks: exit 1, both named, files still written"


def a_capture_of_another_shot() -> str:
    """A capture tied to a row of another shot that reaches into this one is read for her track, not called absent."""
    with tempfile.TemporaryDirectory() as tmp:
        t = made(Path(tmp))
        table = json.loads(t.read_text())
        table["reading"].append({"first": 32, "last": 33, "subject": "a", "present": "yes", "stands": []})
        capture(Path(tmp), "2026-01-04_wide", 20, 14, track=[True] * 14, covered=[True] * 14, parts=[True] * 14, carried=[True] * 14,
                run="rw", sighting={}, grades=[])
        table["rows"].append({"first": 20, "last": 24, "name": "rw", "file": "Video/v/rw_00001.mp4", "subject": "b",
                              "run": f"{Path(tmp) / '2026-01-04_wide'}:rw"})
        out = fc.coverage(table, data=Path(tmp))
        x = [r for r in out["original_while_tracked"] if r["shot"] == [32, 33]]
        assert [(r["frames"], r["track_from"]) for r in x] == [([32, 33], ["2026-01-04_wide"])], (x, out["original_tracking_not_known"])
    return "32-33 read from the capture a row of shot 10-31 ties to"


def looks_and_jobs() -> str:
    """A look in the shots record makes its frames their own kind, out of list 1; a later `seen` takes it back; a laid
    piece's job left_out is said beside its frames and leaves them in list 1 (it may be "left for a later load")."""
    with tempfile.TemporaryDirectory() as tmp:
        t = made(Path(tmp))
        table = json.loads(t.read_text())
        jobf = Path(tmp) / "ra.json"
        jobf.write_text(json.dumps({"job": {"note": "ra, written by mrjob at 2026-01-01T00:00:00Z: ...",
                                            "left_out": [{"subject": "a", "first": 10, "frames": 2, "why": "left for a later load, not ruled out"}]}}))
        ledger = Path(tmp) / "ledger.json"
        ledger.write_text(json.dumps({"renders": {"Video/v/ra_00001.mp4": {"records": [{"job_file": str(jobf)}]}}}))
        shots = Path(tmp) / "shots.json"
        looks = [{"part": "face_and_neck", "first": 25, "last": 26, "is": "not seen", "why": "his back", "by": "mrlook"},
                 {"part": "face_and_neck", "first": 26, "last": 26, "is": "seen", "why": "he turns", "by": "mrlook"}]
        shots.write_text(json.dumps({"shots": [{"first": 10, "last": 31, "who": {"a": {"present": "yes", "seen_looks": looks}}}]}))
        out = fc.coverage(table, data=Path(tmp), ledger=ledger, shots=shots)
        (lo,) = out["left_out_by_a_look"]
        assert lo["frames"] == [25, 25] and lo["who"] == "mrlook" and lo["why"] == "his back", lo
        l1 = {tuple(x["frames"]): x for x in out["original_while_tracked"]}
        assert (26, 26) in l1 and (25, 25) not in l1 and (25, 26) not in l1, sorted(l1)        # 26 taken back by the later `seen`
        said = l1[(10, 11)]["a_jobs_left_out_says"]
        assert said["who"] == "the job of ra, written by mrjob at 2026-01-01T00:00:00Z" and said["frames"] == 2 \
            and said["why"].startswith("left for a later load"), said
        assert any(r["kind"] == "left_out_by_a_look" and r["who"] == "mrlook" for r in out["runs"]), out["runs"]
    return "a look's frames are their own kind with who and why, a later seen takes them back; a job's left_out is said and leaves its frames in list 1"


def ledger_reads_the_same_rule() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made(Path(tmp))
        cap = Path(tmp) / "2026-01-01_ra"
        left = led.left_by_the_load(cap, "a", [12, 19], "ra")
        assert left is not None
        assert left["source_frames"] == [[14, 18]] and left["frames"] == 5, left
        assert left["by_cause"] == {"drop=+nothing saved": [[14, 14]], "floor": [[15, 15]], "grade": [[16, 16]], "nothing saved": [[17, 17]],
                                    "not in the record": [[18, 18]]}, left["by_cause"]
        assert "frame_coverage.py::emptied" in left["read_from"], left
        old = led.left_by_the_load(cap, "a", [12, 19])
        assert old is not None
        assert old["source_frames"] == [[14, 14]] and "parts_emptied_on_source_frames" in old["read_from"], old
    return "with the run: 14-18 by cause, as the tool counts; without: the manifest's drop= list, said"


case("the original while tracked, with or without a piece", original_while_tracked)
case("seams with both render days", seams)
case("emptied inside a piece, by cause", emptied_inside)
case("no capture is not known, not untracked", not_known)
case("a row whose run= does not resolve is red", unresolved_is_red)
case("a capture tied in another shot is read", a_capture_of_another_shot)
case("a look is its own kind; a job's left_out is context", looks_and_jobs)
case("the ledger reads the same rule", ledger_reads_the_same_rule)

if __name__ == "__main__":
    sys.exit(finish())
