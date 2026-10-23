#!/usr/bin/env python3
"""The render history (`bench/render_history.py`) on a made share and records folder: what each row says, where it read it, what it names as unseen.

A share, a records folder, a shots file, a plan, the owner's two answers files, a laid list and a joined cut's sidecar
are made in a temporary folder: empty video files (the frame counts are handed in, nothing is decoded), pictures that
carry a saved graph as ComfyUI writes one, and the job tool's records of a load at each stage it can stop at. Each case is
a load whose files say one thing, with the case that must NOT read that way beside it.

1. **A load that rendered** has one row, read from its own files: the prompt is the song node's `prompt` in the queued
   graph and equals what `bench/what_h3_got.py` reads from the same graph (the two never differ); the padding is the build
   record's `load_against_its_window` (8 real frames before and after) and the same load with no such key says its
   printout's "held for 16 frames"; the zoom is the node's own record beside the file; the preview's commit is the
   preview's `preview.queued.json`; the reference's drawing form is the job's `reference` with the reference record's
   `drawn` (eyes shut by a person's word, by the rule), `passed` and writer commit; the span words are frames; the owner's
   lines are tied by the row (a line copied from the answers file once); it is laid now and was laid in a joined cut;
   and every field carries the file it came from in `sources`.
2. **A load that stopped** has a row with no output file and the furthest chain step with a file, the last line of its
   gate log, and no invented reason; its prompt is the text file beside its job file, said to be that.
3. **A job file with no job folder** is written-and-not-run, said so, with no render field made up.
4. **A render older than the job tool** has the row its saved graph allows, a source for each field it has, and every
   field it has none for in `cannot_see` with where it was looked for; the control is the same render with an earlier
   ledger handed in, which fills the commit and the time from that record and says whose.
5. **What is not a row**: a render of another video, a numbered video with no picture, a re-laid piece (counted apart),
   and a preview's job folder (it is no load).
6. **A row is carried when its records are gone**: with an earlier run's JSONL handed in (`--carry`), a load whose job folder was
   deleted takes its earlier row back, with the owner's lines as they are now; a load whose folder and file are both gone is
   carried as it was; a row the run reads whole is never replaced; the file it was first carried from is kept.
7. **The record is made whole and the same twice**: two runs write the same rows; the CSV is made from the JSONL, has the
   columns `CSV_COLUMNS` names, one row a render, and a prompt with its newlines intact.
8. **It refuses cleanly**: no copy of the video named, a share that is not a folder, a ledger that is not a render
   ledger's JSON, a history that is not a history's JSONL, a video with no data folder each exit with the tool's own words and no traceback.

No video is decoded, no model, no card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_render_history.py
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import render_history as rh  # noqa: E402
import render_ledger as rl  # noqa: E402
import what_h3_got as w  # noqa: E402

COPY = "clip_24p.mov"
SHOTS = {"schema": "h3_shots/1", "video": "clip.mp4", "subjects": ["a"], "span": [0, 99], "shots": [
    {"first": 0, "last": 49, "who": {"a": {"present": "yes"}}}, {"first": 50, "last": 99, "who": {"a": {"present": "yes"}}}]}
PLAN = {"load_prefixes": {"a": ["pre_"], "_by": "a note"}}
PROMPT = "subject_definitions:\n<Subject 1> is the woman in <Picture 1>.\n\nsummary:\nshe sings, " + "x" * 90
WRITTEN = 1767348600.0
QUEUED_AT = "2026-01-02T05:00:00-05:00"          # 10:00 UTC; the file is written half an hour after
SELF = Path(__file__).resolve().parent / "render_history.py"


def picture(path: Path, graph: dict, relaid: dict | None = None, created: str = "2026-01-02 04:31:00") -> None:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    info.add_text("prompt", json.dumps(graph))
    info.add_text("CreationTime", created)
    if relaid is not None:
        info.add_text("relaid", json.dumps(relaid))
    Image.new("RGB", (2, 2)).save(path, pnginfo=info)


def graph(first: int, frames: int, *, loader_first: int | None = None, loader_cap: int | None = None, window: int = 39, copy: str = COPY,
          prompt: str = PROMPT, seed: int = 7, steps: int = 12, motion: str | None = None, region: str | None = "zoomed in on what is replaced") -> dict:
    """A masked render's graph as the job builder shapes one; the source loader may ask for more frames than the load (real padding)."""
    at = (first if loader_first is None else loader_first) - 0.4
    g = {"1": {"class_type": "VHS_LoadVideoFFmpegPath", "inputs": {"video": f"/input/{copy}", "start_time": at / 24.0, "frame_load_cap": loader_cap or frames}},
         "2": {"class_type": "VHS_LoadVideoFFmpegPath", "inputs": {"video": "/m/track.mkv", "start_time": 0.0, "frame_load_cap": frames}},
         "3": {"class_type": "ImageToMask", "inputs": {"image": ["2", 0]}},
         "8": {"class_type": "BasicScheduler", "inputs": {"scheduler": "simple", "steps": steps}},
         "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
         "9": {"class_type": "MiniMaxH3MaskedSource", "inputs": {"frames": ["1", 0], "mask": ["3", 0], "replace": "the wired parts", "grow_pixels": 64,
                                                                 "feather_pixels": 8, "edge": "latent cells", "composite": "whole region",
                                                                 "render_region": "zoomed in on what is replaced", "motion_reference": "a video I wire"}},
         "10": {"class_type": "MiniMaxH3AudioFreezeSong", "inputs": {"source": ["9", 0], "window_frames": window, "context_frames": 90, "seed": seed, "prompt": prompt}},
         "11": {"class_type": "LoadImage", "inputs": {"image": "a.png"}}}
    if region is None:                                        # a graph made before the node had the option
        del g["9"]["inputs"]["render_region"]
    if motion:
        g["6"] = {"class_type": "VHS_LoadVideoFFmpegPath", "inputs": {"video": motion, "start_time": 0.0, "frame_load_cap": frames}}
        g["9"]["inputs"]["motion_video"] = ["6", 0]
    return g


class Made:
    """A share, a records folder and a video's data folder, with renders in them and the frame count each file is said to hold."""

    def __init__(self, root: Path):
        self.root, self.share, self.data = root, root / "output", root / "data"
        self.vdir = self.data / "clip"
        (self.vdir / "verdicts").mkdir(parents=True)
        (self.vdir / "review").mkdir()
        self.counts: dict[Path, int | None] = {}
        (self.vdir / "shots.json").write_text(json.dumps(SHOTS))
        (self.vdir / "plan.json").write_text(json.dumps(PLAN))
        self.laid_lines: list[dict] = []
        self.answers: list[dict] = []
        self.verdicts: list[dict] = []

    def render(self, rel: str, frames: int, g: dict | None, created: str = "2026-01-02 04:31:00", relaid: dict | None = None) -> Path:
        path = self.share / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        os.utime(path, (WRITTEN, WRITTEN))
        self.counts[path] = frames
        if g is not None:
            picture(path.with_suffix(".png"), g, relaid=relaid, created=created)
        return path

    def folder(self, name: str, day: str = "2026-01-02") -> Path:
        f = self.data / f"{day}_{name}_job"
        (f / "chain").mkdir(parents=True, exist_ok=True)
        return f

    def write_job(self, where: Path, name: str, job_extra: dict | None = None, load_extra: dict | None = None, *, text: bool = False) -> None:
        job = {"name": name, "source": f"/input/{COPY}", "first": 30, "frames": 23, "subjects": [{"label": "a", "parts": ["hair", "face_and_neck"]}],
               "texts": {name: {"file": f"{name}.txt", "by": "a hand"}}, "left_out": [], **(job_extra or {})}
        loads = [{"name": name, "first": 30, "frames": 23, "subject": "a", "others": ["held"], "motion": "mesh", "still": "a.png", **(load_extra or {})}]
        where.write_text(json.dumps({"job": job, "defaults": [], "loads": loads}))
        if text:
            (where.parent / f"{name}.txt").write_text(PROMPT + "\nfrom the file")

    def ledger(self, rel: str, **record) -> dict:
        return {"schema": "h3_render_ledger/2", "renders": {rel: {"records": [{"commit": "old1234", "queued_at": "2026-01-01T09:00:00Z", "status": "success",
                                                                               "built_at": "2026-01-01T08:59:00Z", "gate": {"verdict": "clear"}, "looked_at_by": "a reader",
                                                                               "window": 141, "context": 90,
                                                                               "load": {"name": "old", "first": 30, "frames": 23, "subject": "a", "seed": 7}, **record}]}}}

    def history(self, ledgers=None, loads_re=None):
        (self.vdir / "owner_answers.jsonl").write_text("".join(json.dumps(x) + "\n" for x in self.answers))
        (self.vdir / "verdicts" / "verdicts.jsonl").write_text("".join(json.dumps(x) + "\n" for x in self.verdicts))
        (self.vdir / "laid.jsonl").write_text("".join(json.dumps(x) + "\n" for x in self.laid_lines))
        h = rh.History("clip", self.data, self.share, {COPY: 24.0}, repo=self.root, shots=SHOTS, plan=PLAN, verdicts=self.verdicts,
                       answers=rh.jsonl(self.vdir / "owner_answers.jsonl"), laid=self.vdir / "laid.jsonl", ledgers=ledgers,
                       frames_of=lambda p: self.counts.get(p))
        rows, left_out = h.rows(rl.walk(self.share), loads_re)
        return {r["load"] + (" (job file only)" if r["kind"] == "job file only" else ""): r for r in rows}, left_out, rows


def everything(root: Path) -> Made:
    """One of each: a load that rendered with every record, an older one with held padding, a stopped load, a job file alone, a render with no
    record, a render of another video, a re-laid piece, a preview's folder."""
    m = Made(root)
    # a preview and the reference the load read
    preview = m.data / "2026-01-02_shot_c_0030_job"
    preview.mkdir()
    (preview / "preview.json").write_text("{}")
    (preview / "preview.queued.json").write_text(json.dumps({"commit": "pre1234", "queued_at": "2026-01-02T04:00:00-05:00"}))
    ref = m.share / "Video" / "clip" / "references" / "a_30_52" / "a_head_reference_30_52.mp4"
    ref.parent.mkdir(parents=True)
    ref.write_bytes(b"")
    ref.with_suffix(".json").write_text(json.dumps({
        "made_at": "2026-01-02T03:00:00Z", "made_by": "a session", "passed": True, "writer_stamp": {"git_commit": "ref5678", "git_dirty": False},
        "drawn": {"iris": "contrast", "mouth": {"drawn_as": "the lip lines", "closed_lip_line": {"on": True, "closed_frames": [[30, 52]]}},
                  "eyes_shut": {"form": "lowered", "frames": [[30, 52]]},
                  "eye_rings": {"closed_by_the_rule": {"on": True, "frames": [[52, 52]]}}, "frames_left_bare": []}}))
    # good: rendered, real padding 8/8
    rel = "Video/clip/s1/good/good_00001.mp4"
    g = graph(30, 23, loader_first=22, loader_cap=39, motion=str(ref))
    m.render(rel, 23, g)
    f = m.folder("good")
    m.write_job(f / "job.render.json", "good", {"preview": str(preview), "capture": "data/cap_good",
                                               "reference": {"form": "drawn", "said_by": "a", "record": str(ref.with_suffix(".json")),
                                                             "iris": {"form": "contrast", "said_by": "the chain's default",
                                                                      "eyes_shut": {"form": "lowered", "spans": [{"first": 30, "last": 52, "why": "a look"}]}},
                                                             "mouth": {"form": "auto", "closed_lip_line": {"on": True}}, "no_face_on": [{"first": 40, "last": 41, "why": "x"}]},
                                               "carried_words_from": {"from": "good0"}},
                 {"motion_file": str(ref), "seed": 7})
    (f / "good.render.json").write_text(json.dumps(g))
    (f / "good.render.queued.json").write_text(json.dumps({"commit": "abc1234", "queued_at": QUEUED_AT, "status": "success", "files": [rel], "seconds": 120,
                                                          "gate": {"verdict": "clear", "overridden": []}, "look": {"by": "a reader"}, "uncommitted_node_code": [],
                                                          "windows_files": ["good_window_1.mp4", "good_window_1_with_mask.mp4", "good_window_1_zoomed.mp4"],
                                                          "said": {"75": ["rendered -> x; 1 windows [39] with 90-frame context, 39 frames (1.62s) over 1.60s of track; 23 are written"]}}))
    (f / "good.render.built.json").write_text(json.dumps({
        "load": {"name": "good", "first": 30, "frames": 23, "subject": "a", "carried": "held", "others": ["held"], "seed": 7},
        "commit": "abc1234", "built_at": "2026-01-02T04:59:00-05:00", "capture": "data/cap_good", "job_file": "data/clip_inputs/good.json",
        "settings": {"window": {"value": 39}, "context": {"value": 90}, "margin": {"value": 64}, "feather": {"value": 8}, "carried": {"value": "the wired parts"},
                     "seed": {"value": 7}, "render_region": {"value": "zoomed in on what is replaced"}},
        "what_this_clips_preview_measured": {"load_against_its_window": {"frames": 23, "window": 39, "context_before": 8, "context_after": 8,
                                                                         "tossed_after_decode": 16, "held_past_the_load": 0}}}))
    (f / "good.render.build_printed.txt").write_text("  built at abc1234\n    frames 30..52 of the copy, window 39, context 90, padded with 8 real frame(s) before and 8 after (22..60)\n")
    (m.share / Path(rel).parent / "good_00001_zoomed.json").write_text(json.dumps({"box": [10, 20, 110, 95], "zoom": 2.5, "set_by": "the box",
                                                                                   "render_region": "zoomed in on what is replaced"}))
    # held: the same shape of load before the padding key, with the printout's line only
    relh = "Video/clip/s1/held/held_00001.mp4"
    gh = graph(30, 23, window=39)
    m.render(relh, 23, gh)
    fh = m.folder("held")
    (fh / "held.render.queued.json").write_text(json.dumps({"commit": "def5678", "queued_at": QUEUED_AT, "status": "success", "files": [relh]}))
    (fh / "held.render.built.json").write_text(json.dumps({"load": {"name": "held", "first": 30, "frames": 23, "subject": "a"}, "commit": "def5678",
                                                          "settings": {"window": {"value": 39}}}))
    (fh / "held.render.build_printed.txt").write_text("  frames 30..52 of the copy, window 39, context 90, its last frame held for 16 frames with the region open\n")
    # held_b: the structured record of held copies, no printout beside it
    relb = "Video/clip/s1/held_b/held_b_00001.mp4"
    m.render(relb, 23, graph(30, 23, window=39, region=None))
    fb = m.folder("held_b")
    (fb / "held_b.render.queued.json").write_text(json.dumps({"commit": "fed9999", "queued_at": QUEUED_AT, "status": "success", "files": [relb]}))
    (fb / "held_b.render.built.json").write_text(json.dumps({"load": {"name": "held_b", "first": 30, "frames": 23, "subject": "a"}, "commit": "fed9999",
                                                            "what_this_clips_preview_measured": {"load_against_its_window": {"frames": 23, "window": 39, "held_past_the_load": 16}}}))
    # stopped at first: no queue record, a job file elsewhere with a text file beside it, a gate log
    fs = m.folder("pre_stop_0030")
    (fs / "pre_stop_0030.who_is_who.json").write_text("{}")
    (fs / "chain" / "pre_stop_0030.files_first.out").write_text("wrote capture\n")
    (fs / "chain" / "pre_stop_0030.gate_first.out").write_text("27 flag(s)\ngate: blocked, on f001, f002\n")
    (m.data / "inputs").mkdir()
    m.write_job(m.data / "inputs" / "pre_stop_0030.json", "pre_stop_0030", {"capture": "data/2026-01-02_pre_stop_0030"}, text=True)
    # a job file and nothing else; and an older job file of the name `pre_stop_0030`, written and never built, that the next load took the name from
    m.write_job(m.data / "inputs" / "pre_job_0030a.json", "pre_job_0030a")
    (m.data / "inputs_old").mkdir()
    m.write_job(m.data / "inputs_old" / "pre_stop_0030.json", "pre_stop_0030", {"capture": "data/cap_pre_stop_0030_old"})
    # a render with no record at all, from a graph; and one of another video; and a re-laid piece; and one with no picture
    m.render("Video/clip/s1/old/old_00001.mp4", 23, graph(30, 23, window=141, steps=9, prompt="an old prompt " + "y" * 90),
             created="2026-01-01 03:00:00")
    m.render("Video/other/s1/else/else_00001.mp4", 23, graph(30, 23, copy="other.mov"))
    m.render("Video/clip/s1/relaid/relaid_00001.mp4", 23, graph(30, 23), relaid={"from": rel, "sha256": "abc", "by": "s1", "at": "2026-01-02T11:00:00Z", "changed": {}})
    m.render("Video/clip/s1/nopic/nopic_00001.mp4", 23, None)
    # a preview's job folder is no load
    (m.data / "2026-01-02_shot_c_0030_job" / "shot_c_0030.plan.json").write_text("{}")
    # the owner's lines, laid
    m.verdicts = [{"at": "2026-01-02T12:00:00Z", "row": "good", "what": "choice", "chosen": "keep this one", "source": "answers.jsonl:1", "frame": None},
                  {"at": "2026-01-02T12:05:00Z", "row": "good", "what": "words", "words": "the eyes are right", "source": "terminal"},
                  {"at": "2026-01-02T12:05:00Z", "row": "good", "what": "words", "words": "the eyes are right", "source": "owner_answers.jsonl:2"},   # the same line under the answers file's other name
                  {"at": "2026-01-02T16:00:00Z", "row": "page_item_7", "what": "words", "words": "seen on the stacked clip", "source": "terminal",
                   "clip": "data/clip/review/media/good_0030_0052_stacked.mp4"},
                  {"at": "2026-01-02T16:01:00Z", "row": "cut_item", "what": "words", "words": "said of the whole cut", "source": "terminal",
                   "clip": "clip_0_99_joined_2026-01-02T1400Z_VIEW.mp4"}]
    m.answers = [{"at": "2026-01-02T12:00:00Z", "who": "owner", "row": "good", "what": "choice", "chosen": "keep this one"},
                 {"at": "2026-01-02T12:30:00Z", "who": "owner", "row": "held", "what": "words", "words": "wrong, eyes open"}]
    m.laid_lines = [{"schema": "h3_laid/1", "load": "good", "file": rel, "first": 30, "last": 52, "frames": [[30, 52]], "owner": "kept", "added": "2026-01-02T13:00:00Z",
                     "added_by": "s1", "by": "the owner"}]
    cut = m.share / "Video" / "clip" / "day_2026-01-02"
    cut.mkdir(parents=True)
    (cut / "clip_0_99_joined_2026-01-02T1400Z_from_laid.mp4.json").write_text(json.dumps({
        "made_at": "2026-01-02T14:00:00Z", "pieces": [{"name": "good", "first": 30, "last": 52}, {"name": "held", "first": 30, "last": 52}]}))
    return m


def a_load_that_rendered() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        m = everything(Path(tmp))
        rows, left_out, _ = m.history()
        r = rows["good"]
        assert r["kind"] == "chain load" and r["status"] == "rendered" and r["output_file"] == "Video/clip/s1/good/good_00001.mp4", (r["kind"], r["status"], r["output_file"])
        assert (r["subject"], r["span_first"], r["span_last"], r["frames_given"], r["shot_first"], r["shot_last"]) == ("a", 30, 52, 23, 0, 49), r
        assert r["rendered_at"] == "2026-01-02T10:00:00Z" and r["rendered_at_from"] == "the queue record's queued_at" and r["render_seconds"] == 120, (r["rendered_at"], r["rendered_at_from"])
        assert r["png_creation_time"] == "2026-01-02 04:31:00", r["png_creation_time"]
        # the prompt is the song node's, and what_h3_got reads the same text from the same graph
        assert r["prompt"] == PROMPT and r["sources"]["prompt"] == f"good.render.json {w.SONG} prompt", r["sources"]["prompt"]
        # the worktree this runs in need not sit under ComfyUI's custom_nodes, where what_h3_got finds VideoHelperSuite's format table: a one-line stand-in
        fake = Path(tmp) / "load_video_nodes.py"
        fake.write_text("VHSLoadFormats = {'AnimateDiff': {'target_rate': 8, 'dim': (8, 0, 512, 512), 'frames': (4, 1)}}\n")
        was, w.VHS_LOADER_FILE = w.VHS_LOADER_FILE, fake
        try:
            got = w.read(m.data / "2026-01-02_good_job", "good", "render", inputs=m.root / "input", output=m.share, digests=False)
        finally:
            w.VHS_LOADER_FILE = was
        assert got["text"]["text"] == r["prompt"] and got["text"]["sha256"] == r["prompt_sha256"], "what_h3_got.py reads another text than the history"
        assert r["png_graph_prompt_matches"] is True
        assert (r["window_length"], r["padding_before"], r["padding_after"], r["padding_kind"]) == (39, 8, 8, "real source frames"), \
            (r["window_length"], r["padding_before"], r["padding_after"], r["padding_kind"])
        assert r["sources"]["padding_kind"].endswith("load_against_its_window"), r["sources"]["padding_kind"]
        assert (r["seed"], r["steps"], r["sampler"], r["scheduler"], r["context_frames"]) == (7, 12, "euler", "simple", 90), r
        assert r["windows"] == 1 and r["sources"]["windows"] == "queued.json windows_files", (r["windows"], r["sources"].get("windows"))
        assert r["window_lengths"] == [39] and r["sources"]["window_lengths"].startswith("queued.json said"), (r["window_lengths"], r["sources"].get("window_lengths"))
        assert r["zoom"]["box"] == [10, 20, 110, 95] and r["zoom"]["scale"] == 2.5 and r["zoom"]["from"] == "the node's zoomed record beside the render", r["zoom"]
        assert (r["preview"], r["preview_commit"]) == ("data/2026-01-02_shot_c_0030_job", "pre1234"), (r["preview"], r["preview_commit"])
        assert r["commit"] == "abc1234" and r["sources"]["commit"] == "queued.json commit", (r["commit"], r["sources"]["commit"])
        ref = r["reference_form"]
        assert (ref["form"], ref["iris"], ref["mouth"], ref["closed_lip_line"], ref["passed"], ref["commit"], ref["dirty"]) == ("drawn", "contrast", "auto", True, True, "ref5678", False), ref
        assert ref["eyes_shut_by_word"] == {"form": "lowered", "frames": [[30, 52]]} and ref["eyes_shut_by_rule"] == [[52, 52]], ref
        assert r["reference_video"] == "Video/clip/references/a_30_52/a_head_reference_30_52.mp4", r["reference_video"]
        assert r["span_words"] == {"no_face_on": [[40, 41]], "eyes_shut": {"form": "lowered", "spans": [[30, 52]]}}, r["span_words"]
        # a hold written in place from a frame a person named (`word=FIRST-LAST@FRAME`, capture_masked_run.HELD_IN_PLACE): its frames
        # as every hold's, and the frame it is held from beside them; a word without @ says nothing of it
        assert rh.spans("5039-5039@5038+5050-5052") == [[5039, 5039], [5050, 5052]], rh.spans("5039-5039@5038+5050-5052")
        assert rh.span_words({"subjects": [{"word": "5039-5039@5038+5050-5052"}]}) == {
            "hold_part": [[5039, 5039], [5050, 5052]], "hold_part_held_in_place": [[5039, 5039, 5038]]}
        assert rh.span_words({"subjects": [{"word": "5039-5040"}]}) == {"hold_part": [[5039, 5040]]}
        assert r["carried_words_from"] == "good0" and r["keep_outs"] == ["held"] and r["margin"] == 64, (r["carried_words_from"], r["keep_outs"], r["margin"])
        assert r["part"]["parts"] == ["hair", "face_and_neck"] and r["part"]["carried"] == "the wired parts", r["part"]
        assert r["laid"]["now"][0]["frames"] == [[30, 52]] and r["laid"]["in_cuts"] == [{"cut": "clip_0_99_joined_2026-01-02T1400Z_from_laid.mp4", "made_at": "2026-01-02T14:00:00Z",
                                                                                         "frames": [[30, 52]]}], r["laid"]
        said = [(o["said"], o["from"]) for o in r["owner"]]
        assert said == [("keep this one", "verdicts (answers.jsonl:1)"), ("the eyes are right", "verdicts (terminal) = verdicts (owner_answers.jsonl:2)"),
                    ("seen on the stacked clip", "verdicts (terminal), by the clip it names")], said   # each line once, both labels kept; a stacked clip of this load ties, a joined cut does not
        assert r["gate"] == "clear" and r["looked_at_by"] == "a reader" and r["session"] == "s1", (r["gate"], r["looked_at_by"], r["session"])
        assert r["output_frames"] == 23 and r["sources"]["output_frames"] == "ffprobe packets"
        for field, value in r.items():                          # every field with a value says where it came from, bar the ones that are bookkeeping
            if value not in (None, [], {}) and field not in ("schema", "video", "load", "kind", "sources", "cannot_see", "records", "output_files", "png_graph_prompt_matches",
                                                             "png_creation_time", "output_mtime", "reference_folder", "gate_overridden", "uncommitted_node_code", "rendered_at_from"):
                assert field in r["sources"], f"{field} has a value and no source"
        # what no file said is listed with where it was looked for; what was said is not
        assert "seed" not in [x["field"] for x in r["cannot_see"]] and all(x["looked_in"] for x in r["cannot_see"]), r["cannot_see"]
        assert left_out and sum(left_out.values()) == 3, left_out
    return "a load that rendered is read from its own files, field by field, and its prompt is the text what_h3_got.py reads"


def held_padding() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        rows, _, _ = everything(Path(tmp)).history()
        r = rows["held"]
        assert (r["padding_before"], r["padding_after"], r["padding_kind"], r["window_length"], r["frames_given"]) == (0, 16, "held copies of the last frame", 39, 23), r
        assert r["sources"]["padding_kind"] == "build printout", r["sources"]["padding_kind"]
        assert [o["said"] for o in r["owner"]] == ["wrong, eyes open"] and r["laid"]["now"] == [] and len(r["laid"]["in_cuts"]) == 1, (r["owner"], r["laid"])
        g = rows["good"]
        assert g["padding_kind"] != r["padding_kind"], "the two kinds of padding read the same"
    return "padding is the build record's, else the printout's line: real frames and held copies are told apart"


def held_padding_from_the_record() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        rows, _, _ = everything(Path(tmp)).history()
        r = rows["held_b"]
        assert (r["padding_before"], r["padding_after"], r["padding_kind"]) == (0, 16, "held copies of the last frame"), r
        assert r["sources"]["padding_kind"] == "held_b.render.built.json what_this_clips_preview_measured.load_against_its_window", r["sources"]["padding_kind"]
        assert r["window_length"] == 39 and r["loader_frames"] == 23, (r["window_length"], r["loader_frames"])
        # a render that ran on the whole frame has no box, said as that and not as unseen; a zoomed render with no zoom record is unseen
        assert r["zoom"]["box"] is None and r["zoom"]["scale"] == 1.0 and "no render_region" in r["zoom"]["from"] and "zoom" not in {x["field"] for x in r["cannot_see"]}, r["zoom"]
        assert rows["held"]["zoom"] is None and "zoom" in {x["field"] for x in rows["held"]["cannot_see"]}, rows["held"]["zoom"]
    return "held copies are read from the build record's held_past_the_load, with no printout beside it"


def held_across_a_cut() -> str:
    """Context frames across a cut held as copies of the shot's edge frame (2026-10-22, `masked_job_graph.held_context`): read from the
    build record's `context_before_held` / `context_after_held`, and from the printout's line when the record is absent."""
    record = {"what_this_clips_preview_measured": {"load_against_its_window": {
        "frames": 31, "window": 39, "context_before": 4, "context_after": 4, "context_before_held": 4, "context_after_held": 4,
        "tossed_after_decode": 8, "held_past_the_load": 0}}}
    got, where = rh.padding_of(record, "", {}, 31, 4479, 39)
    assert (got["before"], got["after"], got["held"]) == (4, 4, 8) and got["kind"].startswith("held copies of the shot's edge frames across a cut") \
        and "the rest real" not in got["kind"] and "load_against_its_window" in where, (got, where)
    one_side = {"what_this_clips_preview_measured": {"load_against_its_window": {**record["what_this_clips_preview_measured"]["load_against_its_window"],
                                                                                 "context_before_held": 0}}}
    got = rh.padding_of(one_side, "", {}, 31, 4479, 39)[0]
    assert "0 of 4 before, 4 of 4 after" in got["kind"] and "the rest real" in got["kind"], got
    printed = ("    frames 4479..4509 of the copy, window 39, context 90, padded with 4 frame(s) before and 4 after (4475..4513; 4 before "
               "and 4 after lie across a cut from the shot 4479..4509 and are HELD copies of its edge frame")
    got, where = rh.padding_of(None, printed, {}, 31, 4479, 39)
    assert (got["before"], got["after"], got["held"], where) == (4, 4, 8, "build printout") and "across a cut" in got["kind"], (got, where)
    return "held context across a cut is read from the record and from the printout, side by side"


def a_load_that_stopped() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        rows, _, _ = everything(Path(tmp)).history()
        r = rows["pre_stop_0030"]
        assert r["status"] == "stopped before render, at the first step" and r["stopped_after"] == "first" and r["output_file"] is None and r["output_files"] == [], \
            (r["status"], r["stopped_after"], r["output_file"])
        assert r["last_log_line"] == "pre_stop_0030.gate_first.out: gate: blocked, on f001, f002", r["last_log_line"]
        assert r["sources"]["status"].startswith("the furthest chain step with a file"), r["sources"]["status"]
        assert r["prompt"] == PROMPT + "\nfrom the file" and "text file" in r["sources"]["prompt"] and "never reached a queued graph" in r["sources"]["prompt"], r["sources"].get("prompt")
        assert r["rendered_at"] is None and r["recorded_at"] is not None and r["commit"] is None, (r["rendered_at"], r["recorded_at"], r["commit"])
        assert "rendered_at" not in [x["field"] for x in r["cannot_see"]], "a field of a render was asked of a load that never rendered"
    return "a load that stopped says how far its files got and the gate's last line, and its text from the file beside its job file"


def a_name_used_twice() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        m = everything(Path(tmp))
        m.verdicts.append({"at": "2026-01-02T15:00:00Z", "row": "pre_stop_0030", "what": "words", "words": "said of the load that was built", "source": "terminal"})
        rows, _, all_rows = m.history()
        built, old = rows["pre_stop_0030"], rows["pre_stop_0030 (job file only)"]
        assert (built["kind"], old["kind"]) == ("chain load", "job file only") and sum(r["load"] == "pre_stop_0030" for r in all_rows) == 2, [(r["load"], r["kind"]) for r in all_rows]
        assert old["records"]["job_file"] == "data/inputs_old/pre_stop_0030.json" and built["records"]["job_file"] == "data/inputs/pre_stop_0030.json", (old["records"], built["records"])
        assert built["records"]["other_job_files_of_this_name"] == ["data/inputs_old/pre_stop_0030.json"] and old["records"]["other_job_files_of_this_name"] == ["data/inputs/pre_stop_0030.json"]
        assert [o["said"] for o in built["owner"]] == ["said of the load that was built"] and old["owner"] == [], (built["owner"], old["owner"])
    return "two job files of one name are two loads, told apart by the capture they name; the owner's lines go to the load that was built"


def a_job_file_alone() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        rows, _, _ = everything(Path(tmp)).history()
        r = rows["pre_job_0030a (job file only)"]
        assert r["kind"] == "job file only" and r["status"] == "job file written, no chain step ran" and r["output_file"] is None, (r["kind"], r["status"])
        assert (r["span_first"], r["span_last"], r["subject"], r["source_video"]) == (30, 52, "a", COPY), r
        assert r["seed"] is None and r["steps"] is None and r["prompt"] is None, (r["seed"], r["steps"])
        asked = {x["field"] for x in r["cannot_see"]}
        assert "prompt" in asked and not {"seed", "steps", "rendered_at", "commit"} & asked, asked      # a render's fields are not asked of a load that never ran
    return "a job file with no folder is written and not run, with no render field made up"


def an_older_render() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        m = everything(Path(tmp))
        rows, _, _ = m.history()
        r = rows["old"]
        assert r["kind"] == "share render, no job record" and r["status"] == "rendered", (r["kind"], r["status"])
        assert r["prompt"].startswith("an old prompt") and r["sources"]["prompt"] == f"the picture beside the mp4 {w.SONG} prompt", r["sources"].get("prompt")
        assert (r["seed"], r["steps"], r["window_length"], r["source_video"], r["span_first"]) == (7, 9, 141, COPY, 30), r
        assert r["rendered_at"] == "2026-01-01T09:00:00Z" or r["rendered_at_from"].startswith("the picture's CreationTime"), (r["rendered_at"], r["rendered_at_from"])
        assert r["padding_kind"] == "not recorded" and (r["padding_before"], r["padding_after"]) == (0, 118), (r["padding_kind"], r["padding_before"], r["padding_after"])
        missing = {x["field"] for x in r["cannot_see"]}
        assert {"commit", "preview_commit", "reference_form", "span_words", "gate"} <= missing, missing
        assert r["commit"] is None and r["reference_form"] is None and r["span_words"] is None, "a field with no source was filled"
        # the control: an earlier ledger that kept the record fills what it kept, and says whose
        again, _, _ = m.history(ledgers=[("old_ledger.json", m.ledger("Video/clip/s1/old/old_00001.mp4"))])
        o = again["old"]
        assert (o["commit"], o["gate"], o["looked_at_by"]) == ("old1234", "clear", "a reader") and o["sources"]["commit"] == "ledger old_ledger.json", (o["commit"], o["sources"].get("commit"))
        assert o["rendered_at"] == "2026-01-01T09:00:00Z" and "an earlier ledger" in o["rendered_at_from"], (o["rendered_at"], o["rendered_at_from"])
        assert "commit" not in {x["field"] for x in o["cannot_see"]} and "preview_commit" in {x["field"] for x in o["cannot_see"]}
        # a later ledger wins over an earlier one for a file both name
        later, _, _ = m.history(ledgers=[("a.json", m.ledger("Video/clip/s1/old/old_00001.mp4")), ("b.json", m.ledger("Video/clip/s1/old/old_00001.mp4", commit="new9999"))])
        assert later["old"]["commit"] == "new9999", later["old"]["commit"]
    return "an older render has the row its graph allows, each missing field named with where it was looked for, and an earlier ledger fills what it kept"


def what_is_not_a_row() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        rows, left_out, all_rows = everything(Path(tmp)).history()
        assert set(rows) == {"good", "held", "held_b", "pre_stop_0030", "pre_job_0030a (job file only)", "pre_stop_0030 (job file only)", "old"} and len(all_rows) == 7, sorted(rows)
        assert any("re-laid" in k and v == 1 for k, v in left_out.items()), left_out
        assert any("no saved graph" in k and v == 1 for k, v in left_out.items()), left_out
        assert any("other videos" in k and v == 1 for k, v in left_out.items()), left_out
        picked, skipped, _ = everything(Path(tmp + "x")).history(loads_re="^good$")
        assert set(picked) == {"good"} and any("--loads" in k for k in skipped), (sorted(picked), skipped)
    return "other videos, pictureless files, re-laid pieces and a preview's folder are not rows; --loads selects by load name"


def the_record_is_whole_and_the_same_twice() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        m = everything(Path(tmp))
        _, _, rows1 = m.history()
        _, _, rows2 = m.history()
        a, b = Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl"
        rh.write(rows1, a, Path(tmp) / "a.csv")
        rh.write(rows2, b, Path(tmp) / "b.csv")
        assert a.read_bytes() == b.read_bytes() and (Path(tmp) / "a.csv").read_bytes() == (Path(tmp) / "b.csv").read_bytes(), "two runs wrote different bytes"
        lines = a.read_text().splitlines()
        assert len(lines) == len(rows1) == 7 and all(json.loads(x)["schema"] == rh.SCHEMA for x in lines), len(lines)
        assert [json.loads(x)["load"] for x in lines] == [r["load"] for r in sorted(rows1, key=lambda r: (r["rendered_at"] or r["recorded_at"] or "", r["load"], r["output_file"] or ""))]
        with open(Path(tmp) / "a.csv", newline="") as fh:
            table = list(csv.DictReader(fh))
        assert list(table[0].keys()) == list(rh.CSV_COLUMNS) and len(table) == len(lines), (list(table[0].keys()), len(table))
        good = next(x for x in table if x["load"] == "good")
        assert good["prompt"] == PROMPT and "\n" in good["prompt"], "the prompt lost its newlines in the CSV"
        assert good["padding_before"] == "8" and good["zoom_scale"] == "2.5" and good["owner_choices"].endswith("keep this one"), good
        assert json.loads(good["laid"]) == [[[30, 52]]] and good["commit"] == "abc1234", good["laid"]
        c = rh.coverage(rows1)
        assert c["rows"] == 7 and c["by_kind"] == {"chain load": 4, "job file only": 2, "share render, no job record": 1}, c["by_kind"]
        assert c["by_status"]["rendered"] == 4 and "commit" in c["most_often_missing"], c
        assert c["fields_by_source"]["commit"]["queued.json commit"] == 3 and c["most_often_missing"]["commit"] == 1, c["fields_by_source"]["commit"]
    return "the JSONL and the CSV are the same bytes on a second run; the CSV has the named columns and keeps the prompt whole"


def a_row_carried_when_its_records_are_gone() -> str:
    import shutil
    with tempfile.TemporaryDirectory() as tmp:
        m = everything(Path(tmp))
        _, _, before = m.history()
        earlier = Path(tmp) / "before.jsonl"
        rh.write(before, earlier, Path(tmp) / "before.csv")
        shutil.rmtree(m.data / "2026-01-02_held_job")                                   # held: its job folder is gone, its render is on the share
        shutil.rmtree(m.data / "2026-01-02_held_b_job")                                 # held_b: its job folder and its render are both gone
        for suffix in (".mp4", ".png"):
            (m.share / "Video/clip/s1/held_b" / f"held_b_00001{suffix}").unlink()
        m.answers.append({"at": "2026-01-03T09:00:00Z", "who": "owner", "row": "held", "what": "words", "words": "said after the folder went"})
        _, _, now = m.history()
        mine = {(r["load"], r["kind"]): r for r in now}
        assert ("held", "share render, no job record") in mine and ("held_b", "chain load") not in mine and mine[("held", "share render, no job record")]["commit"] is None
        merged, n = rh.carry(now, [x for _, x in rh.jsonl(earlier)], "before.jsonl")
        held = next(r for r in merged if r["load"] == "held")
        assert (held["kind"], held["commit"], held["padding_kind"], held["carried_from"]) == ("chain load", "def5678", "held copies of the last frame", "before.jsonl"), held["kind"]
        assert [o["said"] for o in held["owner"]] == ["wrong, eyes open", "said after the folder went"], held["owner"]       # the owner's lines are as they are now
        b = next(r for r in merged if r["load"] == "held_b")
        assert (b["kind"], b["commit"], b["carried_from"]) == ("chain load", "fed9999", "before.jsonl"), b["carried_from"]
        assert n == 2 and sum(r["carried_from"] is not None for r in merged) == 2 and len(merged) == len(now) + 1, (n, len(merged), len(now))
        assert next(r for r in merged if r["load"] == "good")["carried_from"] is None, "a row this run reads whole was replaced"
        # a carried row carried again keeps the file it was first carried from
        again, _ = rh.carry(now, merged, "second.jsonl")
        assert next(r for r in again if r["load"] == "held_b")["carried_from"] == "before.jsonl"
        # through the command: the carried rows are in the JSONL and counted; a file that is not a history is refused
        out = Path(tmp) / "after.jsonl"
        run = lambda *args: subprocess.run([sys.executable, str(SELF), "--video", "clip", "--data", str(m.data), "--share", str(m.share), "--source", f"{COPY}=24",
                                           "--jsonl", str(out), "--csv", str(Path(tmp) / "after.csv"), *args], capture_output=True, text=True)
        ok = run("--carry", str(earlier))
        assert ok.returncode == 0 and sum(json.loads(x)["carried_from"] == "before.jsonl" for x in out.read_text().splitlines()) == 2, ok.stderr[-300:]
        notmine = Path(tmp) / "x.jsonl"
        notmine.write_text(json.dumps({"schema": "something else"}) + "\n")
        bad = run("--carry", str(notmine))
        assert bad.returncode != 0 and "not a render history's JSONL" in bad.stderr and "Traceback" not in bad.stderr, bad.stderr[-300:]
    return "a load whose records are gone takes its earlier row back with the owner's lines as they are now; a row read whole is never replaced; a file that is not a history is refused"


def it_refuses_cleanly() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        m = everything(Path(tmp))
        run = lambda *args, env=None: subprocess.run([sys.executable, str(SELF), "--video", "clip", "--data", str(m.data), "--jsonl", str(Path(tmp) / "o.jsonl"),
                                                      "--csv", str(Path(tmp) / "o.csv"), *args], capture_output=True, text=True, env={**os.environ, **(env or {})})
        none = run("--share", str(m.share))
        assert none.returncode != 0 and "no copy of the video is named" in none.stderr and "Traceback" not in none.stderr, none.stderr[-300:]
        no_share = run("--source", f"{COPY}=24", "--share", str(Path(tmp) / "nowhere"))
        assert no_share.returncode != 0 and "the share is not a folder" in no_share.stderr and "Traceback" not in no_share.stderr, no_share.stderr[-300:]
        bad = Path(tmp) / "not_a_ledger.json"
        bad.write_text("{}")
        notled = run("--source", f"{COPY}=24", "--share", str(m.share), "--ledger", str(bad))
        assert notled.returncode != 0 and "not a render ledger's JSON" in notled.stderr and "Traceback" not in notled.stderr, notled.stderr[-300:]
        nodata = subprocess.run([sys.executable, str(SELF), "--video", "nope", "--data", str(m.data), "--share", str(m.share), "--source", f"{COPY}=24",
                                 "--jsonl", str(Path(tmp) / "o.jsonl"), "--csv", str(Path(tmp) / "o.csv")], capture_output=True, text=True)
        assert nodata.returncode != 0 and "is not a folder" in nodata.stderr and "Traceback" not in nodata.stderr, nodata.stderr[-300:]
        assert not (Path(tmp) / "o.jsonl").exists(), "a refused run wrote a record"
        ok = run("--source", f"{COPY}=24", "--share", str(m.share), "--loads", "^pre_job_0030a$")
        assert ok.returncode == 0 and (Path(tmp) / "o.jsonl").is_file() and (Path(tmp) / "o.meta.json").is_file(), ok.stderr[-400:]
        meta = json.loads((Path(tmp) / "o.meta.json").read_text())
        assert meta["coverage"]["rows"] == 1 and meta["video"] == "clip", meta["coverage"]
    return "each refusal is the tool's own words with no traceback and no record written; a good run writes the JSONL, the CSV and the meta"


case("a load that rendered", a_load_that_rendered)
case("padding: real frames and held copies", held_padding)
case("held copies, from the record", held_padding_from_the_record)
case("held context across a cut", held_across_a_cut)
case("a load that stopped", a_load_that_stopped)
case("a job file with no job folder", a_job_file_alone)
case("a name used twice", a_name_used_twice)
case("a render older than the job tool, and the ledger that fills it", an_older_render)
case("what is not a row", what_is_not_a_row)
case("the record is whole and the same twice", the_record_is_whole_and_the_same_twice)
case("a row carried when its records are gone", a_row_carried_when_its_records_are_gone)
case("it refuses cleanly", it_refuses_cleanly)

if __name__ == "__main__":
    sys.exit(finish())
