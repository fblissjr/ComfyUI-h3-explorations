#!/usr/bin/env python3
"""`bench/frame_coverage.py` on a made table, made captures, and made build records: each case is a way a frame on the
4-step ladder (`no_capture`, `captured_not_rendered`, `rendered_not_kept`, `kept_and_laid`) could be missed, given the
wrong step or cause, or stripped of its files or canonical words.

No render and no share: the tool reads the table's JSON, the capture records (`manifest.json`, the subject's
`masks__<by>.npz`, `part_grades.json`, the run's `region.npz`), and the build records (`*_job/*.render.built.json` and
`*.render.queued.json`), so the fixture is those files, written here.

The made shot is 10-31, subject `a` (`unsure`). Piece `ra` (capture one, 10-29) lays 12-19; piece `rb` (capture two,
20-24) lays 20-24. Inside `ra` the node carried nothing on 14 (`drop=`, its saved part empty too), 15 (the floor), 16
(the grade off its classes), 17 (nothing saved) and 18 (her track empty there too); 13 is held. Outside the pieces she is
tracked on 10-11 and 25-26 in capture one, her track is empty on 27, capture one covers 28-29 with no track of her there
(step 1: `no_capture`), and untied capture `2026-01-03_untied` holds her track on 30-31 (step 2: `captured_not_rendered`,
naming `2026-01-03_untied/subjects/a/masks__p.npz`).

1. **Step 2 (`captured_not_rendered`) and Step 4 (`emptied_inside`)**: 10-11, 25-26 (tied) and 30-31 (untied) with no
   piece, naming their mask files, and 14, 15, 16, 17 each with its cause; 18 is not in it (her track is empty) and 13 is
   not (held, carried).
2. **Seams**: 19/20, `ra` meeting `rb`, each with its render day.
3. **Emptied inside a piece**: the four with their causes, and nothing else.
4. **Step 1 (`no_capture`) vs Step 2 (`captured_not_rendered` from untied capture)**: 28-29 come back as `no_capture`
   with why and `2026-01-01_ra/manifest.json`; 30-31 are read from untied `2026-01-03_untied` as `captured_not_rendered`
   naming `2026-01-03_untied/subjects/a/masks__p.npz`; 18 and 27 (her track 0 px) are their own list; the unsure shot is
   read and said.
5. **A row whose run= does not resolve** is a hard failure and the tool exits 1, and its frames are not known, not
   "no piece laid".
6. **A capture tied to a row of another shot** that reaches this shot's frames is read for her track there, and a
   frame whose piece's mask is not known while her track is known is `carried_not_known`, not `no_capture`.
7. **A look and a job's left_out** (the lead's ruling of 2026-10-16, as built): frames the shots record says are not
   seen are "left out by a look", with who and why, and out of list 1; a later `seen` takes them back; a laid
   piece's job left_out is said beside its frames and leaves them in list 1.
8. **The ledger reads the same rule**: `render_ledger.left_by_the_load` given the run gives 14-18 by cause; with no
   run it falls back to the manifest's `drop=` list and says so.
9. **Laid but barely changed; a region far below its own size**: 21-22 at 0.125 barely changed; 23 at 1 px of its median
   32 too small; `by_shot` ranks `remake` and `capture` separately.
10. **No clip check, or another table's, is refused**: exit 2, nothing written.
11. **The 4-step ladder (`no_capture`, `captured_not_rendered`, `rendered_not_kept`, `kept_and_laid`) and canonical
    words / `clip_gap_line`**: an unkept render in `*_job/*.render.built.json` + `.queued.json` covering 25-26 promotes
    25-26 from Step 2 (`captured_not_rendered`) to Step 3 (`rendered_not_kept`) naming its clip, windows, and build
    record; every frame in `shots[0]` carries `step`, `step_name`, and `files`; `run_words`, `chip_words_of`, and
    `clip_gap_line` format per-person counts and spans in the check's own words.
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
    (d / "manifest.json").write_text(json.dumps({"clip": "c.mov", "first_frame": first, "frames": n, "runs": runs, "size": [W, H],
                                                 "subjects": [{"label": "a", "sightings": [{"by": "p", **sighting}]}]}))
    return d


def clip_check(root: Path, table: dict, changed: dict | None = None) -> dict:
    """The assembled clip's check beside the table, as the assembler writes it (`<clip>.mp4.check.json`), its table the
    table's rows; each piece changed 32 px a frame (its whole carried region, at a crop the capture's size) but where
    `changed` says ({name: {frame: px}}). The table's assembly command names the clip."""
    clip = root / "clip.mp4"
    table["assembly"] = {"command": f"py bench/assemble_delivery.py --table t.txt --out {clip}"}
    pieces = {}
    for r in table["rows"]:
        piece = f"/out/{r['file']}"
        a, b = (min(r["first"], pieces[piece]["frames"][0]), max(r["last"], pieces[piece]["frames"][1])) if piece in pieces else (r["first"], r["last"])
        pieces[piece] = {"frames": [a, b], "changed_px_per_frame": [(changed or {}).get(r["name"], {}).get(f, 32) for f in range(a, b + 1)]}
    (root / "clip.mp4.check.json").write_text(json.dumps({"table": [{"piece": f"/out/{r['file']}", "frames": [r["first"], r["last"]]} for r in table["rows"]],
                                                         "the_loader's_crop_of_the_source": {"width": W, "height": H, "x": 0, "y": 0},
                                                         "regions": {"pieces": pieces}}))
    (root / "t.json").write_text(json.dumps(table))
    return table


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
    clip_check(root, table)
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
                ((25, 26), None, "original: no piece laid"),
                ((30, 31), None, "original: no piece laid")]
        assert got == want, got
        cnr = [(tuple(x["frames"]), x["tied"], x["mask_files"]) for x in out["captured_not_rendered"]]
        assert cnr == [((10, 11), True, ["2026-01-01_ra/subjects/a/masks__p.npz"]),
                       ((25, 26), True, ["2026-01-01_ra/subjects/a/masks__p.npz"]),
                       ((30, 31), False, ["2026-01-03_untied/subjects/a/masks__p.npz"])], cnr
        st = dict(zip(out["shots"][0]["frames"], out["shots"][0]["state"]))
        assert st[13] == "carried (held from a neighbouring frame)" and st[18].startswith("original: emptied") and st[27] == "original: no piece laid", st
        assert out["captured_not_rendered"][0]["track_px"] == {"least": 32, "median": 32, "most": 32}, out["captured_not_rendered"][0]
    return "captured_not_rendered (tied on 10-11, 25-26; untied on 30-31 naming masks__p.npz) and emptied frames each with its cause; empty track and held frame left out"


def seams() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out = fc.coverage(json.loads(made(Path(tmp)).read_text()), data=Path(tmp))
        (s,) = out["seams"]
        assert s["at"] == [19, 20] and (s["before"]["piece"], s["before"]["day"], s["after"]["piece"], s["after"]["day"]) == ("ra", "01/01", "rb", "01/02"), s
        kinds = [(r["first"], r["last"], r["kind"], r["pieces"], r["cause"]) for r in out["runs"]]
        assert kinds == [(10, 11, "captured_not_rendered", [], "2026-01-01_ra/subjects/a/masks__p.npz"),
                         (14, 14, "emptied_inside", ["ra"], "drop=+nothing saved"),
                         (15, 15, "emptied_inside", ["ra"], "floor"),
                         (16, 16, "emptied_inside", ["ra"], "grade"),
                         (17, 17, "emptied_inside", ["ra"], "nothing saved"),
                         (18, 18, "original_her_track_empty", ["ra"], "original: emptied inside the piece (not in the record)"),
                         (19, 20, "seam", ["ra", "rb"], None),
                         (25, 26, "captured_not_rendered", [], "2026-01-01_ra/subjects/a/masks__p.npz"),
                         (27, 27, "original_her_track_empty", [], "original: no piece laid"),
                         (28, 29, "no_capture", [], "a capture covers this frame and holds no track of her on it (2026-01-01_ra/manifest.json)"),
                         (30, 31, "captured_not_rendered", [], "2026-01-03_untied/subjects/a/masks__p.npz")], kinds
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
        (u,) = out["no_capture"]
        assert u["frames"] == [28, 29] and u["step"] == 1 and u["kind"] == "no_capture" \
            and u["why"] == "a capture covers this frame and holds no track of her on it (2026-01-01_ra/manifest.json)" \
            and u["files"] == {"captures": [], "mask_files": [], "captures_without_track": ["2026-01-01_ra/manifest.json"]}, u
        v = next(x for x in out["captured_not_rendered"] if x["frames"] == [30, 31])
        assert v["step"] == 2 and not v["tied"] and v["mask_files"] == ["2026-01-03_untied/subjects/a/masks__p.npz"], v
        i = out["shots"][0]["frames"].index(28)
        assert out["shots"][0]["track_px"][i] is None and out["shots"][0]["step"][i] == 1, out["shots"][0]
        j = out["shots"][0]["frames"].index(30)
        assert out["shots"][0]["track_px"][j] == 32 and out["shots"][0]["step"][j] == 2, out["shots"][0]
        assert [x["frames"] for x in out["original_her_track_empty"]] == [[18, 18], [27, 27]], out["original_her_track_empty"]
        assert out["shots"][0]["present"] == "unsure" and all(x["present"] == "unsure" for x in out["runs"] if x["kind"] != "seam"), out["runs"]
    return "28-29 Step 1 (no_capture) naming 2026-01-01_ra/manifest.json; 30-31 Step 2 (captured_not_rendered) from untied capture naming masks__p.npz; empty track its own list"


def unresolved_is_red() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        t = made(Path(tmp))
        table = json.loads(t.read_text())
        table["rows"].append({"first": 25, "last": 26, "name": "gone", "file": "Video/v/gone.mp4", "subject": "a", "run": f"{Path(tmp) / 'nope'}:gone"})
        table["rows"].append({"first": 27, "last": 27, "name": "norun", "file": "Video/v/norun.mp4", "subject": "a",
                              "run": f"{Path(tmp) / '2026-01-01_ra'}:norun"})
        clip_check(Path(tmp), table)
        r = subprocess.run([sys.executable, str(TOOL), "--table", str(t)], capture_output=True, text=True)
        assert r.returncode == 1 and "FAIL row gone: run=" in r.stdout and "no run 'norun'" in r.stdout, r.stdout + r.stderr
        out = fc.coverage(json.loads(t.read_text()), data=Path(tmp))
        st = dict(zip(out["shots"][0]["frames"], out["shots"][0]["state"]))
        assert st[25] == "not known: the laid piece gone has a run= that does not resolve" and st[27].startswith("not known: the laid piece norun"), st
        # her track is known on 25-26 (32 px) and 27 (0 px): the piece's mask is what is not known there, not her tracking
        assert [(x["frames"], x["track_px"]["median"]) for x in out["carried_not_known"]] == [([25, 26], 32), ([27, 27], 0)], out["carried_not_known"]
        assert not any(x["frames"][0] <= 26 and x["frames"][1] >= 25 for x in out["no_capture"]), out["no_capture"]
        md = (Path(tmp) / "frame_coverage_t.md").read_text()
        cov_json = json.loads((Path(tmp) / "frame_coverage_t.json").read_text())
        assert "1: no capture; 2: captured, not rendered; 3: rendered, not kept; 4: kept and laid" in cov_json["cannot_see"], cov_json["cannot_see"]
        assert (Path(tmp) / "frame_coverage_t.json").is_file() and md.count("## FAILED") == 1 \
            and "## Step 1. Nothing captured (`no_capture`)" in md, md
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
        out = fc.coverage(clip_check(Path(tmp), table), data=Path(tmp))
        x = [r for r in out["captured_not_rendered"] if r["shot"] == [32, 33]]
        assert [(r["frames"], r["track_from"], r["mask_files"]) for r in x] == [
            ([32, 33], ["2026-01-04_wide"], ["2026-01-04_wide/subjects/a/masks__p.npz"])
        ], (x, out["no_capture"])
    return "32-33 read from the capture a row of shot 10-31 ties to, naming 2026-01-04_wide/subjects/a/masks__p.npz"


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


def barely_changed_and_too_small() -> str:
    """The lead's orders of 2026-10-16 (806 and 805 in the 2003Z cut): a laid piece that changed under `BARELY_CHANGED` of the
    region its run carried is `laid_but_barely_changed`, read from the clip's check; a carried region under
    `FAR_BELOW_ITS_OWN_SIZE` of the piece's own median is `region_far_below_its_own_size`; a frame the job emptied stays
    emptied; a fully changed piece gives neither."""
    with tempfile.TemporaryDirectory() as tmp:
        t = made(Path(tmp))
        table = json.loads(t.read_text())
        base = fc.coverage(table, data=Path(tmp))
        assert not base["laid_but_barely_changed"] and not base["region_far_below_its_own_size"], "a fully changed piece gave a kind"
        # THE CASE THAT GOES RED when the ranking mixes what a capture answers with what a remake answers (the lead, 2026-10-16)
        assert base["by_shot"] == {"remake": [{"shot": [10, 31], "frames": 10, "subjects": ["a"], "by_kind": {"captured_not_rendered": 6, "emptied_inside": 4}}],
                                   "capture": [{"shot": [10, 31], "frames": 2, "subjects": ["a"], "by_kind": {"no_capture": 2}}]}, base["by_shot"]
        # rb changed 4 of its 32 carried px on 21 and 22; ra changed nothing on 14-18, where its job emptied her part
        clip_check(Path(tmp), table, {"rb": {21: 4, 22: 4}, "ra": {f: 0 for f in range(14, 19)}})
        out = fc.coverage(table, data=Path(tmp))
        (x,) = out["laid_but_barely_changed"]
        # THE CASE THAT GOES RED when a laid piece that changed almost nothing of its region is called carried
        assert x["frames"] == [21, 22] and x["piece"] == "rb" and x["changed_share"]["least"] == 0.125, x
        # BOTH COUNTS AT THE CANVAS (Gemini's D17): the loader's crop of the source is not a scale of either; a check built at the source's
        # size says a crop twice the canvas, and one built at the canvas's says none
        held = json.loads((Path(tmp) / "clip.mp4.check.json").read_text())
        for crop in ({"width": 2 * W, "height": 2 * H, "x": 0, "y": 0}, None):
            if crop is None:
                held.pop("the_loader's_crop_of_the_source", None)
            else:
                held["the_loader's_crop_of_the_source"] = crop
            (Path(tmp) / "clip.mp4.check.json").write_text(json.dumps(held))
            (y_,) = fc.coverage(table, data=Path(tmp))["laid_but_barely_changed"]
            # THE CASE THAT GOES RED when the carried region is scaled by the crop, or a check with no crop is not read
            assert y_["changed_share"]["least"] == 0.125, (crop, y_["changed_share"])
        run = next(r for r in out["runs"] if r["kind"] == "laid_but_barely_changed")
        assert run["cause"] == "0.125 to 0.125 of its carried region changed (under 0.3)", run
        assert [r["first"] for r in out["runs"] if r["kind"] == "emptied_inside"] == [14, 15, 16, 17], "an emptied frame changed kind"
        assert {"shot": [10, 31], "frames": 12, "subjects": ["a"], "by_kind": {"captured_not_rendered": 6, "emptied_inside": 4,
                "laid_but_barely_changed": 2}} == out["by_shot"]["remake"][0], out["by_shot"]
        # rb's carried region is one px on 23, against its median of 32
        z = Path(tmp) / "2026-01-02_rb" / "runs" / "rb" / "region.npz"
        m = np.zeros((5, H, W), bool)
        m[:, 2:6, 3:11] = True
        m[3] = False
        m[3, 0, 0] = True
        np.savez(z, carried=np.packbits(m, axis=-1), read=np.ones(5, bool), region=np.zeros((5, 1, 1), bool))
        clip_check(Path(tmp), table)
        out = fc.coverage(table, data=Path(tmp))
        (y,) = out["region_far_below_its_own_size"]
        # THE CASE THAT GOES RED when a scrap of a region, the original round it, is called carried
        assert y["frames"] == [23, 23] and y["carried_px"] == [1] and y["its_median_px"] == 32, y
        assert not out["laid_but_barely_changed"], out["laid_but_barely_changed"]
    return "rb's 21-22 at 0.125 barely changed; ra's emptied 14-17 stay emptied; rb's 23 at 1 px of its median 32 too small; a fully changed piece neither; by shot counts them"


def no_check_refused() -> str:
    """The gap check reads what each piece changed from the clip's check, so it is refused without it, and on a check of
    another table; never run and silently skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        t = made(Path(tmp))
        table = json.loads(t.read_text())
        (Path(tmp) / "clip.mp4.check.json").rename(Path(tmp) / "kept.json")
        try:
            fc.coverage(table, data=Path(tmp))
        except fc.Refused as e:
            # THE CASE THAT GOES RED when the gap check runs with no clip check
            assert "clip.mp4.check.json: missing" in str(e), e
        else:
            raise AssertionError("ran with no clip check")
        r = subprocess.run([sys.executable, str(TOOL), "--table", str(t)], capture_output=True, text=True)
        assert r.returncode == 2 and "REFUSED" in r.stdout and not (Path(tmp) / "frame_coverage_t.json").exists(), r.stdout + r.stderr
        (Path(tmp) / "kept.json").rename(Path(tmp) / "clip.mp4.check.json")
        table["rows"][1]["last"] = 23
        try:
            fc.coverage(table, data=Path(tmp))
        except fc.Refused as e:
            assert "is not of this table" in str(e) and "rb_00001.mp4 20-24" in str(e), e
        else:
            raise AssertionError("ran on a check of another table")
        table["rows"][1]["last"] = 24
        r = subprocess.run([sys.executable, str(TOOL), "--table", str(t), "--check", str(Path(tmp) / "clip.mp4.check.json")], capture_output=True, text=True)
        assert r.returncode == 0 and json.loads((Path(tmp) / "frame_coverage_t.json").read_text())["table_sha256"], r.stdout + r.stderr
    return "no check: refused, exit 2, nothing written; a check of another table refused naming the rows; --check read, the table's sha256 recorded"


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


def four_step_ladder_and_words() -> str:
    """C129 and C131: every frame gets its furthest step reached on the 4-step ladder (1: no_capture; 2: captured_not_rendered;
    3: rendered_not_kept from *_job/*.render.built.json + *.render.queued.json; 4: kept_and_laid), naming the files;
    canonical words and clip_gap_line format per-person counts and spans in the check's own words."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        t = made(root)
        table = json.loads(t.read_text())
        job_dir = root / "2026-01-02_rc_job"
        job_dir.mkdir(parents=True)
        win_dir = root / "Video" / "v" / "windows"
        win_dir.mkdir(parents=True)
        (win_dir / "rc_window_00_00001.mp4").write_bytes(b"win")
        (job_dir / "rc.render.built.json").write_text(json.dumps({
            "capture": str(root / "2026-01-01_ra"),
            "load": {"name": "rc", "subject": "a", "first": 25, "frames": 2, "window": 33, "context": 8},
            "settings": {"window": {"value": 33}, "context": {"value": 8}},
            "windows_folder": str(win_dir),
            "output_folder": str(root / "Video" / "v"),
        }))
        (job_dir / "rc.render.queued.json").write_text(json.dumps({
            "files": ["Video/v/rc_00001.mp4"],
            "windows_files": ["Video/v/windows/rc_window_00_00001.mp4"],
        }))
        out = fc.coverage(table, data=root)
        sh = out["shots"][0]
        step_by_f = dict(zip(sh["frames"], sh["step"]))
        name_by_f = dict(zip(sh["frames"], sh["step_name"]))
        files_by_f = dict(zip(sh["frames"], sh["files"]))
        # Step 1 on 28-29
        assert step_by_f[28] == 1 and name_by_f[28] == "no_capture", (step_by_f[28], name_by_f[28])
        assert files_by_f[28]["captures_without_track"] == ["2026-01-01_ra/manifest.json"], files_by_f[28]
        # Step 2 on 10-11 (tied) and 30-31 (untied)
        assert step_by_f[10] == 2 and name_by_f[10] == "captured_not_rendered", (step_by_f[10], name_by_f[10])
        assert files_by_f[10]["mask_files"] == ["2026-01-01_ra/subjects/a/masks__p.npz"], files_by_f[10]
        assert step_by_f[30] == 2 and name_by_f[30] == "captured_not_rendered", (step_by_f[30], name_by_f[30])
        assert files_by_f[30]["mask_files"] == ["2026-01-03_untied/subjects/a/masks__p.npz"], files_by_f[30]
        # Step 3 on 25-26 (rendered in rc, not in the table)
        assert step_by_f[25] == 3 and name_by_f[25] == "rendered_not_kept", (step_by_f[25], name_by_f[25])
        assert files_by_f[25]["clips"] == ["Video/v/rc_00001.mp4"], files_by_f[25]
        assert files_by_f[25]["windows"] == ["Video/v/windows/rc_window_00_00001.mp4"], files_by_f[25]
        assert files_by_f[25]["built_files"] == ["2026-01-02_rc_job/rc.render.built.json"], files_by_f[25]
        (rnk,) = out["rendered_not_kept"]
        assert rnk["frames"] == [25, 26] and rnk["clips"] == ["Video/v/rc_00001.mp4"] \
            and rnk["windows"] == ["Video/v/windows/rc_window_00_00001.mp4"] \
            and rnk["built_files"] == ["2026-01-02_rc_job/rc.render.built.json"], rnk
        # Step 4 on 12-24 (kept and laid)
        assert step_by_f[12] == 4 and name_by_f[12] == "kept_and_laid", (step_by_f[12], name_by_f[12])
        assert files_by_f[12] == {
            "piece": "ra",
            "piece_file": "Video/v/ra_00001.mp4",
            "capture": "2026-01-01_ra",
            "region_file": "2026-01-01_ra/runs/ra/region.npz",
            "mask_file": "2026-01-01_ra/subjects/a/masks__p.npz",
        }, files_by_f[12]
        assert out["kept_and_laid"] and out["kept_and_laid"][0]["frames"] == [12, 12], out["kept_and_laid"]
        # Canonical words stamped on every run, and clip_gap_line in the check's own words
        assert all("words" in r and "chip_words" in r for r in out["runs"]), out["runs"]
        line_all = fc.clip_gap_line(out, 10, 31)
        assert line_all == (
            "Gaps on 10-31 — a: no capture: 2 (28-29); captured, not rendered: 4 (10-11, 30-31); "
            "rendered, not kept: 2 (25-26); emptied inside a piece: 4 (14-17)"
        ), line_all
        assert fc.clip_gap_line(out, 20, 24) == "", "whole clip span should produce empty gap line"
    return "steps 1..4 with files (untied mask_files, unkept render clips/windows/built_files, laid region_file); canonical words and clip_gap_line verified"


def real_form_run_resolves() -> str:
    """C151: a row whose `run=` has the real relative form `data/<day>_<load>:<load>` resolves against `data` and
    `Path.cwd()` even when `REPO` (e.g. a worktree `HERE.parent`) holds no `data/` directory."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "data"
        capture(data_dir, "2026-01-01_ra", 10, 5, track=[True] * 5, covered=[True] * 5, parts=[True] * 5,
                carried=[True] * 5, run="ra", sighting={}, grades=[])
        table = {
            "rows": [{"first": 10, "last": 14, "name": "ra", "file": "Video/v/ra_00001.mp4", "subject": "a",
                      "run": "data/2026-01-01_ra:ra"}],
            "reading": [{"first": 10, "last": 14, "subject": "a", "present": "yes",
                         "stands": [{"render": "ra", "written": "2026-01-01T18:00:00Z"}]}],
        }
        clip_check(root, table)
        old_repo = fc.REPO
        try:
            fc.REPO = root / "worktree_without_data"
            out = fc.coverage(table, data=data_dir)
        finally:
            fc.REPO = old_repo
        # THE CASE THAT GOES RED when resolve() only checks REPO / folder and fails on relative run=data/<day>_<load>:<load>
        assert not out["failures"], out["failures"]
        c = fc.counts(out)
        assert c["kept_and_laid"]["frames"] == 5 and c["kept_and_laid"]["carried_frames"] == 5, c
        r = subprocess.run(
            [sys.executable, str(TOOL), "--table", str(root / "t.json")],
            cwd=str(root), capture_output=True, text=True,
        )
        assert r.returncode == 0 and "FAIL" not in r.stdout, r.stdout + r.stderr
    return "run=data/<day>_<load>:<load> resolves against data= and cwd when REPO has no data/ folder"


case("the original while tracked, with or without a piece", original_while_tracked)
case("seams with both render days", seams)
case("emptied inside a piece, by cause", emptied_inside)
case("no capture is not known, not untracked", not_known)
case("a row whose run= does not resolve is red", unresolved_is_red)
case("a capture tied in another shot is read", a_capture_of_another_shot)
case("a look is its own kind; a job's left_out is context", looks_and_jobs)
case("the ledger reads the same rule", ledger_reads_the_same_rule)
case("laid but barely changed; a region far below its own size", barely_changed_and_too_small)
case("no clip check, or another table's, is refused", no_check_refused)
case("the 4-step ladder, files, and canonical words", four_step_ladder_and_words)
case("a relative run=data/<day>_<load>:<load> resolves from a worktree or data=", real_form_run_resolves)

if __name__ == "__main__":
    sys.exit(finish())
