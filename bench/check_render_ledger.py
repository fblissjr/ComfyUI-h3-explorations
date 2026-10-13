#!/usr/bin/env python3
"""The render ledger (`bench/render_ledger.py`) on a made share: what each state of a render's making takes.

A share, a records folder, a shots file and a plan are made in a temporary folder: empty video files (the frame
counts are handed in, so nothing is decoded), pictures that carry a saved graph as ComfyUI writes one, and the job
tool's two records of a queued render. Each case is a render whose files say one thing, with the case that must
NOT be read that way beside it.

1. **A render its record agrees with** is in the section of every shot its load covers and of its subject; says
   what it replaces from the capture's own part classes; says its reference video from the records beside the
   video, followed back to the body model, and where the chain ends; says what the text calls `<Video 1>`; is
   said to be laid where the table lays it; and carries the owner's lines by its row and by its file, a line the
   verdicts file copied from the answers file once and one it did not copy from the answers file. A line said
   of a JOINED clip at stated frames is under the render that clip's table laid there, printed as said of the
   joined clip, and one whose frames the table gave to another piece stays loose, named as a viewing copy when
   the page has an item for it. A record's still is looked for only when the input folder is handed in.
2. **A file that does not hold its record's frames disagrees**, with both figures; a load that continued a stored
   window says the record cannot say which source frames the file holds (the case of 2026-10-12 this tool was
   written to name). A run that did not end `success`, a saved graph that asks for other frames than the
   record, and a file written before its record was queued each disagree, and the agreeing render (the control)
   says none of them.
3. **A queue record from before the job tool named its files** is tied by the build record's output folder and
   the load's name, in a state of its own, and its file's count against the load is still said.
4. **A render with no record is listed, not dropped**: its form from the saved graph alone, a reference made
   inside the graph said as that, its subject from the capture folder its mask sits in, read through the plan's
   names. A note in the plan (a key that begins with `_`) is nobody's prefix. A record tied to a file by its
   folder is not listed among the records with no render.
5. **What is not a render of this video is counted and left out**: a render of another video, a numbered video
   whose graph has no sampler, one with no picture beside it, and the files beside a render (its mask picture,
   its audio copy, its windows). A record whose render is gone is listed.
6. **The page tells a render once** and points at it from its other sections; **the summary holds no private
   path and none of the owner's words**.

No video is decoded, no model, no card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_render_ledger.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import render_ledger as led  # noqa: E402

COPY = "clip_24p.mov"
SHOTS = {"schema": "h3_shots/1", "video": "clip.mp4", "subjects": ["a", "b"], "span": [0, 99], "shots": [
    {"first": 0, "last": 49, "who": {"a": {"present": "yes"}, "b": {"present": "no"}}},
    {"first": 50, "last": 99, "who": {"a": {"present": "yes"}, "b": {"present": "unsure"}}}]}
PLAN = {"aliases": {"singer": "a", "_by": "a note"}, "load_prefixes": {"b": ["bee_"], "_by": "a note"}}
QUEUED_AT, WRITTEN = "2026-01-02T10:00:00+00:00", 1767348600.0      # the file half an hour after the queue


def picture(path: Path, graph: dict) -> None:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    info.add_text("prompt", json.dumps(graph))
    Image.new("RGB", (2, 2)).save(path, pnginfo=info)


def graph(first: int, frames: int, mask: str, *, copy: str = COPY, parts: str | None = None, motion: str | None = None,
          motion_node: str | None = None, replace: str = "the wired parts", text: str = "", continue_from: str = "") -> dict:
    """A masked render's graph as the job builder shapes one: the loader set a fraction of a frame early."""
    load = lambda video, at=0.0, cap=frames: {"class_type": "VHS_LoadVideoFFmpegPath", "inputs": {"video": video, "start_time": at, "frame_load_cap": cap}}
    g = {"1": load(f"/input/{copy}", (first - 0.4) / 24.0), "2": load(mask), "3": {"class_type": "ImageToMask", "inputs": {"image": ["2", 0]}},
         "9": {"class_type": "MiniMaxH3MaskedSource", "inputs": {"frames": ["1", 0], "mask": ["3", 0], "replace": replace, "grow_pixels": 16,
                                                                 "motion_reference": "a video I wire" if motion or motion_node else "none"}},
         "10": {"class_type": "MiniMaxH3AudioFreezeSong", "inputs": {"source": ["9", 0], "window_frames": 141, "context_frames": 90, "seed": 7,
                                                                     "prompt": text or "x" * 90, "continue_from": continue_from}},
         "11": {"class_type": "LoadImage", "inputs": {"image": "a.png"}}}
    if parts:
        g["4"], g["5"] = load(parts), {"class_type": "ImageToMask", "inputs": {"image": ["4", 0]}}
        g["9"]["inputs"]["parts"] = ["5", 0]
    if motion:
        g["6"] = load(motion)
        g["9"]["inputs"]["motion_video"] = ["6", 0]
    if motion_node:
        g["7"] = {"class_type": motion_node, "inputs": {"images": ["1", 0]}}
        g["9"]["inputs"]["motion_video"] = ["7", 0]
    return g


class Made:
    """A share and a records folder with renders in them, and the frame count each file is said to hold."""
    def __init__(self, root: Path):
        self.root, self.share, self.records, self.counts = root, root / "output", root / "data", {}
        self.capture = self.records / "2026-01-01_cap"
        (self.capture / "subjects" / "a").mkdir(parents=True)
        (self.capture / "manifest.json").write_text(json.dumps({"subjects": [
            {"label": "a", "sightings": [{"by": "p", "part_is_made_of": ["Face_Neck", "Hair", "Upper_Lip"]}]}]}))
        for name in ("track.mkv", "parts_held.mkv"):
            (self.capture / "subjects" / "a" / name).write_bytes(b"")
        self.mask, self.parts = (str(self.capture / "subjects" / "a" / n) for n in ("track.mkv", "parts_held.mkv"))

    def render(self, rel: str, frames: int | None, g: dict | None, written: float = WRITTEN) -> Path:
        path = self.share / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        os.utime(path, (written, written))
        self.counts[path] = frames
        if g is not None:
            picture(path.with_suffix(".png"), g)
        return path

    def record(self, name: str, load: dict, files: list[str] | None, *, status: str | None = "success", folder: str | None = None,
               text: str | None = None, queued_at: str = QUEUED_AT) -> None:
        job = self.records / f"2026-01-02_{name}_job"
        job.mkdir(parents=True, exist_ok=True)
        built = {"load": {"name": name, "subject": "a", "carried": "held", **load}, "commit": "abc1234", "capture": str(self.capture),
                 "files": {"track": [self.mask, 0], "parts": [self.parts, 0], "others": [[str(self.root / "gone.mkv"), 0]]},
                 "settings": {"window": {"value": 141}, "context": {"value": 90}}, "output_folder": folder, "uncommitted_node_code": []}
        if text is not None:
            (job / f"{name}.txt").write_text(text)
            built["text"] = {"file": str(job / f"{name}.txt")}
        (job / f"{name}.render.built.json").write_text(json.dumps(built))
        queued = {"commit": "abc1234", "queued_at": queued_at, "gate": {"verdict": "clear", "overridden": []}, "look": {"by": "a reader"}}
        if files is not None:
            queued.update(status=status, files=files)
        (job / f"{name}.render.queued.json").write_text(json.dumps(queued))

    def ledger(self, **more) -> dict:
        given = dict(table_rows=[], verdicts=[], answers=[], items=[], spots=None, inputs=None)
        given.update(more)
        return led.ledger(SHOTS, PLAN, self.share, self.records, self.root, {COPY: 24.0}, given["table_rows"], given["verdicts"], given["answers"],
                          given["items"], frames_of=lambda p: self.counts.get(p), alias=lambda label: PLAN["aliases"].get(label, label),
                          spots=given["spots"], inputs=given["inputs"])


def _reference(made: Made) -> str:
    """A reference video with the records its tools write: a body model as a preview saved it, cut to the head,
    then the face points drawn on the cut."""
    folder = made.share / "Video" / "clip" / "s1" / "good_inputs"
    folder.mkdir(parents=True)
    for name in ("mesh_00001.mp4", "cut.mp4", "ref.mp4"):
        (folder / name).write_bytes(b"")
    picture(folder / "mesh_00001.png", {"1": {"class_type": "SaveVideo", "inputs": {}}})
    (folder / "cut.json").write_text(json.dumps({"box": [10, 20, 110, 95], "scale": [2.5, 2.5], "made_from": {"reference_video": str(folder / "mesh_00001.mp4")}}))
    (folder / "ref.json").write_text(json.dumps({"line_width_px": 1.5, "parts_left_out": ["mouth"], "a_dot_at_every_point": False, "made_by": "s1",
                                                 "made_from": {"body_model_video": str(folder / "cut.mp4")}}))
    return str(folder / "ref.mp4")


TEXT = ("subject_definitions:\n<Subject 1> is the woman of <Picture 1>, who follows the figure in <Video 1>.\n"
        "<Subject 2> is the movements of the figure in <Video 1>: a closer view of a plain grey body model.\n\nsummary:\n<Subject 1> performs <Subject 2>.\n")


def a_render_its_record_agrees_with() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        ref = _reference(made)
        rel = "Video/clip/s1/good/good_00001.mp4"
        made.render(rel, 60, graph(30, 60, made.mask, parts=made.parts, motion=ref, text=TEXT))
        made.render("Video/clip/s1/good/good_00001_with_mask.mp4", 60, None)
        made.record("good", {"first": 30, "frames": 60, "motion": "mesh", "motion_file": ref, "still": "a.png", "seed": 7}, [rel], text=TEXT)
        verdicts = [{"at": "2026-01-03T00:00:00Z", "row": "good", "what": "words", "words": "the eyes are right", "source": "answers.jsonl:2", "render": None},
                    {"at": "2026-01-03T00:01:00Z", "row": "r__s1__other", "what": "choice", "chosen": "keep this one", "source": "the terminal", "render": rel},
                    {"at": "2026-01-03T00:02:00Z", "row": "a_stacked_copy", "what": "words", "words": "not tied", "source": "the terminal", "render": None}]
        answers = [(2, {"at": "2026-01-03T00:00:00Z", "who": "owner", "row": "good", "what": "words", "words": "the eyes are right"}),
                   (3, {"at": "2026-01-03T00:03:00Z", "who": "owner", "row": "good", "what": "words", "words": "said on the page later"}),
                   (4, {"at": "2026-01-03T00:04:00Z", "who": "a session (test of the page)", "row": "good", "what": "words", "words": "not the owner's"})]
        out = made.ledger(table_rows=[{"first": 30, "last": 89, "piece": str(made.share / rel)}], verdicts=verdicts, answers=answers)
        r = out["renders"][rel]
        assert r["making"] == led.MAKING[0] and r["records"][0]["disagrees"] == [] and r["records"][0]["names_the_file"], r["records"]
        assert (r["span"], r["span_read_from"], r["file_frames"], r["subject"], r["session_folder"]) == ([30, 89], "the record's load", 60, "a", "s1"), r
        assert r["mask_picture"] == {"file": "Video/clip/s1/good/good_00001_with_mask.mp4", "frames": 60}, r["mask_picture"]
        # in both shots its load covers, for its subject; b has a section on the shot it is unsure on, with nothing in it
        assert [(s["first"], s["subject"], s["renders"]) for s in out["sections"]] == [(0, "a", [rel]), (50, "a", [rel]), (50, "b", [])], out["sections"]
        form = r["form"]
        assert form["replaces"].startswith("face and hair (the capture's part is made of Face_Neck, Hair, Upper_Lip)"), form["replaces"]
        # the same as fields, for a tool: the region in the plan's word, and which steps the reference's records hold
        assert form["region"] == "face and hair" and form["reference"]["has"] == {"cut_to_the_head": True, "face_points": True, "lid_marks": False}, form
        assert out["schema"] == led.SCHEMA and out["share"] == str(made.share)
        # the tool's own command as the ledger keeps it: one line a shell takes as it stands, an argument with a space quoted
        kept = led.command_of(["--share", "/a share", "--source", "c.mov=23.976"], python="py")
        assert kept["line"] == "py bench/render_ledger.py --share '/a share' --source c.mov=23.976" and kept["argv"][1] == "/a share", kept
        says = form["reference"]["says"]
        assert "cut to one box round the head ([10, 20, 110, 95], enlarged 2.5 times), then the face points drawn as lines (left out: mouth)" in says \
            and "the chain ends at mesh_00001.mp4" in says and form["reference"]["made_by"] == "s1", says
        assert form["text_says_the_video_is"].startswith("<Subject 1> is the woman") and "a closer view of a plain grey body model" in form["text_says_the_video_is"]
        assert r["laid_in_the_table"] == [[30, 89]] and out["counts"]["laid_in_the_table"] == 1
        said = [(o["said"], o["from"]) for o in r["owner"]]
        assert said == [("the eyes are right", "verdicts (answers.jsonl:2)"), ("keep this one", "verdicts (the terminal)"),
                        ("said on the page later", "answers.jsonl:3")], said
        assert [x["row"] for x in out["not_placed"]["verdict_rows_tied_to_no_render_file"]] == ["a_stacked_copy"], out["not_placed"]
        assert out["not_placed"]["verdict_rows_tied_to_no_render_file"][0]["why"].startswith("a row with no render's file"), out["not_placed"]
        # one of the record's three inputs is gone, and that is a count, not a disagreement; the still is not looked for unasked
        on = r["records"][0]["inputs_on_disk"]
        assert (on["named"], [Path(m).name for m in on["missing"]], on["capture_folder"]) == (3, ["gone.mkv"], True), on
        assert (on["still"], on["still_is_a_file"]) == ("a.png", None), on
        (made.root / "input").mkdir()
        (made.root / "input" / "a.png").write_bytes(b"")
        assert made.ledger(inputs=made.root / "input")["renders"][rel]["records"][0]["inputs_on_disk"]["still_is_a_file"] is True
        # A LINE SAID OF A JOINED CLIP at stated frames is under the render that clip's table laid there, as that and
        # no more; a spot on frames the table laid another piece on, or nothing, is under no render and stays loose
        spots = [{"clip": "the joined clip", "table": "t.txt", "rows": [{"first": 30, "last": 59, "piece": str(made.share / rel)},
                                                                        {"first": 60, "last": 89, "piece": "/elsewhere/other_00001.mp4"}],
                  "spots": [{"row": "spot_1", "first": 40, "last": 45}, {"row": "spot_2", "first": 70, "last": 75}]}]
        on_clip = [{"at": "2026-01-01T00:00:00Z", "row": "spot_1", "what": "words", "words": "the hand is wrong here", "source": "the page", "render": None},
                   {"at": "2026-01-01T00:01:00Z", "row": "spot_2", "what": "choice", "chosen": "bothers me", "source": "the page", "render": None}]
        again = made.ledger(verdicts=on_clip, spots=spots, items=[{"id": "spot_2", "title": "a viewing copy of 70 to 75"}])
        (one,) = again["renders"][rel]["owner"]
        assert (one["said"], one["on_the_joined_clip"], one["at_frames"], one["through_the_table"]) == ("the hand is wrong here", "the joined clip", [40, 45], "t.txt"), one
        assert "ON THE JOINED CLIP the joined clip at 40-45, through the table `t.txt` (said of the joined clip there, not of this render alone)" in \
            "\n".join(led.render_lines(again["renders"][rel]))
        assert again["not_placed"]["verdict_rows_tied_to_no_render_file"] == [
            {"row": "spot_2", "why": "a stacked or joined viewing copy; which render it shows is in the item's title", "title": "a viewing copy of 70 to 75"}]
    return ("a render its record agrees with: its sections, what it replaces by the capture's classes, its reference by the records beside "
            "it, the text's own words, the table, and the owner's lines once each")


def a_file_against_its_record() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        place = lambda name: f"Video/clip/s1/{name}/{name}_00001.mp4"
        made.render(place("fine"), 20, graph(10, 20, made.mask))
        made.record("fine", {"first": 10, "frames": 20}, [place("fine")])
        # the load of 2026-10-12: it loads its frames from an earlier first frame, keeps a stored window's, writes the rest
        made.render(place("went_on"), 15, graph(30, 40, made.mask, continue_from="/output/x_window_1.safetensors"))
        made.record("went_on", {"first": 30, "frames": 40, "continue_from": "/output/x_window_1.safetensors"}, [place("went_on")])
        made.render(place("failed"), 20, graph(10, 20, made.mask))
        made.record("failed", {"first": 10, "frames": 20}, [place("failed")], status="error")
        made.render(place("other_frames"), 20, graph(12, 20, made.mask))
        made.record("other_frames", {"first": 10, "frames": 20}, [place("other_frames")])
        made.render(place("older"), 20, graph(10, 20, made.mask), written=WRITTEN - 86400.0)
        made.record("older", {"first": 10, "frames": 20}, [place("older")])
        out = made.ledger()["renders"]
        assert out[place("fine")]["making"] == led.MAKING[0] and out[place("fine")]["records"][0]["disagrees"] == [], out[place("fine")]["records"]
        for name, words in (("went_on", ("40 frames from 30", "the file holds 15", "continues a stored window", "does not say which source frames")),
                            ("failed", ("status is 'error'",)), ("other_frames", ("asks for 20 frames from 12", "is 20 from 10")),
                            ("older", ("before the record was queued", "not that run's"))):
            r = out[place(name)]
            said = " | ".join(r["records"][0]["disagrees"])
            assert r["making"] == led.MAKING[1] and len(r["records"][0]["disagrees"]) == 1 and all(w in said for w in words), (name, r["making"], said)
    return ("a file that does not hold its record's frames, a load that continued a stored window, a run that did not end in success, a graph "
            "that asks for other frames and a file older than its queue each disagree, each in its own words, and the agreeing one says none")


def an_older_record_and_no_record() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        old = "Video/s0/batch/early/early_00001.mp4"
        made.render(old, 34, graph(10, 33, made.mask))
        made.record("early", {"first": 10, "frames": 33}, None, folder=str(made.share / "Video/s0/batch/early"))
        # beside it, a file of another load's name in the same folder is NOT that record's
        made.render("Video/s0/batch/early/early_b_00001.mp4", 33, graph(10, 33, made.mask))
        bare = "Video/s0/by_hand/whole_00001.mp4"
        made.render(bare, 50, graph(50, 50, str(made.root / "data/2026-01-01_cap/subjects/singer/track.mkv"), replace="whole subject", motion_node="BodyModelRender"))
        named = "Video/s0/by_hand/bee_walk_00001.mp4"
        made.render(named, 50, graph(50, 50, "/somewhere/track.mkv", replace="whole subject"))
        nobody = "Video/s0/by_hand/a_take_00001.mp4"        # begins with a letter of the plan's note, which is no prefix
        made.render(nobody, 50, graph(50, 50, "/somewhere/track.mkv", replace="whole subject"))
        out = made.ledger()
        r = out["renders"][old]
        assert r["making"] == led.MAKING[2] and r["records"][0]["names_the_file"] is False and r["session_folder"] == "s0", (r["making"], r["records"])
        assert r["records"][0]["disagrees"] == ["the record's load is 33 frames from 10; the file holds 34"], r["records"][0]["disagrees"]
        assert out["renders"]["Video/s0/batch/early/early_b_00001.mp4"]["making"] == led.MAKING[3], "a file of another load's name took the record"
        r = out["renders"][bare]
        assert (r["making"], r["subject"], r["subject_read_from"], r["span"]) == (led.MAKING[3], "a", "the capture folder its mask file sits in", [50, 99]), r
        assert r["form"]["replaces"] == "whole person [the saved graph and the capture its mask file sits in]", r["form"]["replaces"]
        assert r["form"]["region"] == "whole" and r["form"]["reference"]["has"] == led.NO_STEPS, r["form"]
        assert r["form"]["reference"]["says"].startswith("made in the graph by BodyModelRender: not a file"), r["form"]["reference"]
        assert (out["renders"][named]["subject"], out["renders"][named]["subject_read_from"]) == ("b", "the load name's prefix, by the plan")
        assert out["renders"][nobody]["subject"] is None and out["renders"][nobody]["subjects"] == []
        by = {(s["first"], s["subject"]): s for s in out["sections"]}
        assert by[(50, "a")]["renders"] == [bare] and by[(50, "b")]["renders"] == [named], by
        assert by[(50, "subject not known")]["renders"] == [nobody] and (0, "subject not known") not in by, by
        assert not any(s["subject"].startswith("_") for s in out["sections"]), "a note in the plan was read as a subject"
        counts = out["counts"]
        assert (counts["renders"], counts[led.MAKING[2]], counts[led.MAKING[3]]) == (5, 1, 4), counts
        assert out["not_placed"]["records_whose_render_is_not_on_the_share"] == [], "a record tied to a file by its folder was listed as having no render"
    return ("a queue record that names no file is tied by its build record's folder and load name, in a state of its own, and its count is "
            "still said; a render with no record is listed with its form from the graph, a reference made in the graph said as that, and "
            "its subject from its mask's folder, the plan's prefix, or nobody")


def what_is_not_a_render_of_this_video() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        made.render("Video/s0/other/dance_00001.mp4", 10, graph(0, 10, made.mask, copy="another_clip.mov"))
        made.render("Video/s0/preview/mesh_00001.mp4", 10, {"1": {"class_type": "SaveVideo", "inputs": {}}})
        made.render("Video/s0/loose/loose_00001.mp4", 10, None)
        made.render("Video/s0/x/x_00001.mp4", 10, graph(0, 10, made.mask))
        for beside in ("x_00001_with_mask.mp4", "x_00001-audio.mp4", "x_windows/x_window_1_00001.mp4", "x_00001_zoomed.mp4", "notes.mp4"):
            made.render(f"Video/s0/x/{beside}", 10, graph(0, 10, made.mask))
        made.record("vanished", {"first": 0, "frames": 10}, ["Video/s0/vanished/vanished_00001.mp4"])
        out = made.ledger()
        assert list(out["renders"]) == ["Video/s0/x/x_00001.mp4"], list(out["renders"])
        n = out["not_placed"]
        assert n["renders_of_other_videos_by_the_copy_loaded"] == {"another_clip.mov": 1}, n
        assert n["numbered_videos_whose_graph_is_no_render_of_a_video"] == {"no sampler in the saved graph: not a render": 1}, n
        assert n["numbered_videos_with_no_saved_graph_and_no_record"] == ["Video/s0/loose/loose_00001.mp4"], n
        assert [g["load"]["name"] for g in n["records_whose_render_is_not_on_the_share"]] == ["vanished"], n
        # a day's floor leaves out what was written before it, and nothing after
        assert list(made.ledger().get("renders")) == list(led.ledger(SHOTS, PLAN, made.share, made.records, made.root, {COPY: 24.0}, [], [], [], [],
                                                                      since="2026-01-02", frames_of=lambda p: 10)["renders"])
        assert led.ledger(SHOTS, PLAN, made.share, made.records, made.root, {COPY: 24.0}, [], [], [], [], since="2026-01-03", frames_of=lambda p: 10)["renders"] == {}
    return ("a render of another video, a numbered video with no sampler in its graph and one with no picture are counted and left out; a "
            "mask picture, an audio copy, a window and a zoomed picture are not candidates; a record whose render is gone is listed")


def the_page_and_the_summary() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        made = Made(Path(tmp))
        rel = "Video/clip/s1/good/good_00001.mp4"
        made.render(rel, 60, graph(30, 60, made.mask, parts=made.parts))
        made.record("good", {"first": 30, "frames": 60}, [rel])
        words = "the owner's own sentence about this render"
        out = made.ledger(verdicts=[{"at": "2026-01-03T00:00:00Z", "row": "good", "what": "words", "words": words, "source": "the terminal", "render": None}])
        text = led.page(out, "Made for the check.", "a_table.txt")
        assert text.count("  - ITS MAKING: recorded, the file agrees") == 1 and text.count('told in full under "0-49, a"') == 1, text
        assert words in text and "## 50-99, b (the shots file says present: unsure)" in text and "No render covers these frames for this subject." in text
        short = led.summary(out, "Made for the check.")
        assert words not in short and str(made.root) not in short and "| 0-49 | a | 1 | 1 | 0 | 0 | 0 | 0 | 1 |" in short, short
        assert all(word not in text.lower() for word in (" best ", " better than this", " worse ")), "the page ranks a render"
    return ("a render on two shots is told in full once and pointed at from the other; a shot nobody rendered says so; the summary has "
            "the counts and neither the owner's words nor a path of the machine")


case("a render its record agrees with", a_render_its_record_agrees_with)
case("a file against its record: the frames, the outcome, the graph, the time", a_file_against_its_record)
case("a record that names no file, and no record at all", an_older_record_and_no_record)
case("what is not a render of this video", what_is_not_a_render_of_this_video)
case("the page tells a render once; the summary holds no private word", the_page_and_the_summary)

if __name__ == "__main__":
    sys.exit(finish())
