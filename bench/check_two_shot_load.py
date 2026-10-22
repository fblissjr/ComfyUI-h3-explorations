#!/usr/bin/env python3
"""Two neighbouring shots as one load (`bench/two_shot_load.py`) on a made video: what it lays end to end, writes and refuses.

A video of two neighbouring shots (100-139 and 140-189) is made in a temporary folder: its analysis folder, its
shots record and plan, a finished preview of each shot with REAL mask videos (the lane's own writer, a different
picture on every frame), shot tables and pose tables, and an ordinary load on each shot written by
`bench/shot_to_load.py`'s own `load_job`, with a small reference video each. The machine is handed in.

1. **Videos laid end to end decode equal to their sources**, frame for frame, read back here by the lane's
   reader and not by the tool's own hash; a part that does not hold its frames, parts of two sizes and a file
   that is there are refused; and a joined file that does not decode equal is REMOVED and refused.
2. **A joined shot table** holds the first table's shots as they were and the second's with every frame number
   moved on, by hand for each field `shot_table.py` numbers by frame, and the cut where the two meet; the
   lane's own reader of a table cut to a stretch (`who_is_who_across_shots.cut_to`) gets each source's shots
   back. Tables of two sizes are refused. **A joined pose table** the same; two that do not meet are refused.
3. **The two loads are held to each other**: another seed (the words behind a `--seed`, `defaults.seed_by`, are a record and not a difference: two
   loads of one seed asked for at two times are made one load), margin, subject, still or reference form, a load
   zoomed on a person's word (the chain's default zoom is set aside and the load rendered on the whole frame, said), keep-outs, loads that do not meet at the cut, a second load that is not on the next shot, a preview
   that is not of its whole shot, and a load with no reference made are each refused by name.
4. **The job step** lays every subject's track, part and class map end to end with each frame equal to its
   source's, writes the joined preview's folder SAID TO BE JOINED, the joined reference and the job file; the
   job names `two_shots` with the record's cut, its why and who; `masked_job_graph.check_load` takes its load;
   the chain's `read_job`, `preview_files` and `mask_specs` read it as any load, every mask from the joined
   preview's first frame, and its `first` step hands the capture the joined files. `--dry` writes nothing; a
   second run, no `--why`, and two shots too long for one window are refused. A load made again (`--remake`)
   reads the joined reference that is there only while it decodes equal to the two loads' own, and is refused
   when one of them was made again since.
5. **The builder lets the one named cut through and nothing else** (`masked_job.texts_for` on a capture that
   holds the joined table): the job's text with a `[Shot 1]` and a `[Shot 2]` is taken; the same text in the
   same job WITHOUT `two_shots` is refused as a text of two shots; one header, or a third, is refused.
6. **The latent steps at the cut** are the planner's own (`loop_plan.split_steps`): every first frame the tool
   lists as putting the cut on a step's edge gives no split step and every other first frame of the cycle gives
   one; a part on one side only is the capture's blocking flag and on both its iffy one, by the names
   `bench/capture_masked_run.py` gives them.
7. **`plan-cuts`**: the node's own cuts in its plan against the job's; exit 0 when they are the same list, 1
   when the node found none or another.
8. **A piece on two record shots** is laid on both shots' rows of the coverage table (`shot_coverage.coverage`),
   each row cut to its own frames, and the page's item carries the whole load's frames (`load_chain.page_item`).

No server, no card. It needs `ffmpeg`.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_two_shot_load.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "workflows"))
sys.path.insert(0, str(HERE.parent))
from _lib import case, finish  # noqa: E402
from _lib.frames import probe, stream, write_mask_video  # noqa: E402

import clip_folder  # noqa: E402
import load_chain as lc  # noqa: E402
import masked_job  # noqa: E402
import masked_job_graph as graphs  # noqa: E402
import shot_coverage  # noqa: E402
import shot_to_load as stl  # noqa: E402
import two_shot_load as tsl  # noqa: E402
import who_is_who_across_shots as wiw  # noqa: E402

SIZE = (32, 24)
#: The made video's two shots, and the two loads: the first begins where the cut (140) is on a latent step's edge.
SHOT_A, SHOT_B, A_FIRST, CUT = (100, 139), (140, 189), 106, 140


class Machine:
    """The machine a step is handed: an empty queue, a `run` that records, and the video's voice tables."""
    def __init__(self, root: Path):
        self.host, self.dry, self.python, self.pose_model = "nowhere:0", False, "PY", "pose-model"
        self.output, self.inputs, self.data = root / "output", root / "input", root / "data"
        self.ran = []

    def queue(self):
        return (0, 0)

    def voice(self, source_name):
        return {"per_frame": self.data / "voice.csv", "spans": self.data / "spans.json", "frames": [0, 259]}

    def run(self, ctx, label, argv, stop=True):
        self.ran.append((label, [str(x) for x in argv]))
        return 0

    def said(self, label: str) -> list[str]:
        return next(argv for name, argv in self.ran if name == label)


def refused(fn, *words: str) -> str:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except (lc.Refused, masked_job.Refused) as e:
        assert all(w in str(e) for w in words), f"refused without saying {words}: {e}"
        return str(e)
    raise AssertionError(f"not refused; expected {words}")


def pictures(first: int, frames: int, salt: int) -> np.ndarray:
    """[frames, h, w] of bool, a different picture on every frame and for every `salt`: one lit block whose place is the source frame's own."""
    w, h = SIZE
    out = np.zeros((frames, h, w), bool)
    for i in range(frames):
        n = (first + i) * 7 + salt * 3
        out[i, n % (h - 4):n % (h - 4) + 4, (n // 3) % (w - 4):(n // 3) % (w - 4) + 4] = True
    return out


def read_back(path) -> np.ndarray:
    w, h, n = probe(str(path))
    return np.stack([frame > 127 for frame in stream(str(path), (w, h), 0, n, vf=f"scale={w}:{h}:flags=neighbor")])


def a_reference(path: Path, first: int, frames: int) -> None:
    """A small h264 video, a different picture a frame: what a head reference is to the tool, a video of the load's frames."""
    w, h = SIZE
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "18", str(path)], stdin=subprocess.PIPE)
    enc.communicate((pictures(first, frames, 5).astype(np.uint8) * 200).tobytes())
    assert enc.returncode == 0, f"ffmpeg could not write {path}"


def a_table(frames: int, shown: int) -> dict:
    """A shot table of one shot in the tracker's own form, with a value under every field it numbers by frame."""
    return {"table": "h3 shot table", "version": wiw.TABLES, "frames": frames, "size": list(SIZE), "pick": "largest", "pick_frame": shown, "match": 0.9,
            "cuts": [], "track_score_trouble": [],
            "shots": [{"shot": 1, "first_frame": 0, "last_frame": frames - 1, "frames": frames, "shown_frame": shown,
                       "people": [{"person": 1, "box": [1, 2, 3, 4], "has_head": True}],
                       "subject": {"state": "picked", "person": 1, "seed_frame": shown}, "seeded": {"frame": shown, "box": [1.0, 2.0, 3.0, 4.0], "person": 1},
                       "frames_with_subject": frames - 2, "frames_without_subject": [[3, 4]],
                       "regained": [{"lost_from_frame": 3, "seed_frame": 5, "detections": 2}], "probes_after_a_loss": [{"frame": 4, "detections": 1}],
                       "gallery_frames": [shown], "looks": [{"frame": 6, "people": [{"person": 1, "box": [1, 2, 3, 4]}]}], "frames_on_others": [[7, 8]],
                       "track_score": [0.5] * frames}],
            "gallery": {"frames": [shown], "signatures": [[[0.1]]], "whole": [[0.2]], "handed_in": 0, "pick_probes": []}}


def a_pose(first: int, frames: int) -> dict:
    return {"schema": "h3_body_pose_table/1", "image_size": list(SIZE), "first_source_frame": first,
            "frames": [{"frame": i, "source_frame": first + i, "people": []} for i in range(frames)]}


class Made:
    """The made video: its analysis folder, its record and plan, the two previews as finished, and the two ordinary loads."""
    def __init__(self, root: Path, labels=("a", "b")):
        self.root, self.m, self.labels = root, Machine(root), labels
        for d in ("input", "output", "data/v", "jobs"):
            (root / d).mkdir(parents=True)
        (root / "input" / "copy.mov").write_bytes(b"")
        about = lambda name: {"name": name, "bytes": 0, "frames": 260, "rate": 24.0, "size": [64, 48],   # noqa: E731
                              "head_sha256": clip_folder.head_hash(root / "input" / "copy.mov")}
        clip_folder.write_manifest(root / "data/v", {"source": about("v.mp4"), "derived": about("copy.mov")})
        who = lambda present: {k: {"present": "yes" if k in present else "no", "how": "a person, on a picture", "by": "a person", "frames_on": []} for k in "abc"}   # noqa: E731
        self.shots = {"schema": "h3_shots/1", "video": "v.mp4", "source_frames": 260, "copy": "copy.mov", "span": [0, 259], "subjects": ["a", "b", "c"],
                      "shots": [{"first": 0, "last": 99, "who": who("")}, {"first": SHOT_A[0], "last": SHOT_A[1], "who": who(labels)},
                                {"first": SHOT_B[0], "last": SHOT_B[1], "who": who(labels)}, {"first": 190, "last": 259, "who": who("a")}]}
        self.plan = {"schema": "h3_plan/1", "video": "v.mp4", "subjects": {k: {"region": "face", "by": "a person", "still": f"{k}.png", "parts": ["face_and_neck"]} for k in "abc"}}
        self.shots_file, self.plan_file, self.jobs = root / "shots.json", root / "plan.json", root / "jobs"
        self.shots_file.write_text(json.dumps(self.shots))
        self.plan_file.write_text(json.dumps(self.plan))
        self.source = str(root / "input" / "copy.mov")
        self.previews = {first: self.preview_ran(first, last) for first, last in (SHOT_A, SHOT_B)}
        self.a = self.load(SHOT_A, "a_fp_0106", first=A_FIRST, frames=SHOT_A[1] - A_FIRST + 1, why="so that the cut is on a latent step's edge")
        self.b = self.load(SHOT_B, "a_fp_0140")

    def read(self):
        return stl._load(str(self.shots_file), str(self.plan_file))

    def preview_ran(self, first: int, last: int) -> tuple[dict, Path]:
        """A finished preview of one shot: its job folder, real mask videos of every subject, a shot table and a pose table each."""
        shots, plan = self.read()
        job = stl.preview_job(shots, plan, stl.shot_of(shots, first), "v", self.source, "the session", now="now")
        name, frames = job["job"]["name"], last - first + 1
        folder = self.root / "data" / f"2026-01-02_{name}_job"
        folder.mkdir(parents=True)
        (folder / "job.json").write_text(json.dumps(job))
        out, listed = self.root / "output" / job["job"]["output"] / name, []
        out.mkdir(parents=True)
        for s in job["job"]["subjects"]:
            label = s["label"]
            for salt, kind in enumerate(tsl.KINDS):
                write_mask_video(out / f"{name}_{label}_{kind}_00001.mkv", pictures(first, frames, salt + 10 * "abc".index(label)))
                listed.append(f"{job['job']['output']}/{name}/{name}_{label}_{kind}_00001.mkv")
            (out / f"{name}_{label}_shots_00001_shots.json").write_text(json.dumps(a_table(frames, 9)))
            (out / f"{name}_{label}_pose_table.json").write_text(json.dumps(a_pose(first, frames)))
        (folder / "preview.queued.json").write_text(json.dumps({"status": "success", "files": listed}))
        return job["job"], folder

    def load(self, shot: tuple[int, int], name: str, **more) -> Path:
        """An ordinary load on a shot, as `shot_to_load.py job` and `reference` leave it: its job file, and its reference made."""
        shots, plan = self.read()
        pjob, folder = self.previews[shot[0]]
        saved = json.loads((folder / "preview.queued.json").read_text())["files"]
        job = stl.load_job(shots, plan, stl.shot_of(shots, shot[0]), "a", pjob, str(folder), "v", self.m.output, "the session", "2026-01-03", name, now="now",
                           source=self.source, preview_files=saved, **more)
        ref = Path(job["loads"][0]["motion_file"])
        ref.parent.mkdir(parents=True, exist_ok=True)
        if not ref.exists():
            a_reference(ref, job["loads"][0]["first"], job["loads"][0]["frames"])
        job["job"]["reference"] = {"form": "drawn", "why": "the form", "said_by": "the session", "made": "then", "record": str(ref.with_suffix(".json"))}
        return self.write(name, job)

    def write(self, name: str, job: dict) -> Path:
        path = self.jobs / f"{name}.json"
        path.write_text(json.dumps(job, indent=1))
        return path

    def changed(self, which: Path, name: str, change) -> Path:
        """A copy of an ordinary load's job file under another name with one thing changed by `change(job dict)`."""
        job = json.loads(which.read_text())
        change(job)
        return self.write(name, job)

    def args(self, **more) -> argparse.Namespace:
        given = dict(a=str(self.a), b=str(self.b), shots=str(self.shots_file), folder=str(self.jobs), by="the session", why="one face across the cut, for one pair",
                     name=None, remake=False, day="2026-01-03")
        given.update(more)
        return argparse.Namespace(**given)

    def job(self, **more) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tsl.step_job(self.args(**more), self.m)
        return out.getvalue()


def videos_laid_end_to_end() -> str:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        one, two = pictures(0, 7, 1), pictures(50, 5, 2)
        write_mask_video(tmp / "one.mkv", one)
        write_mask_video(tmp / "two.mkv", two)
        held = tsl.lay_end_to_end([(tmp / "one.mkv", 7), (tmp / "two.mkv", 5)], tmp / "out" / "joined.mkv")
        back = read_back(tmp / "out" / "joined.mkv")
        assert back.shape[0] == 12 and (back[:7] == one).all() and (back[7:] == two).all(), "the joined file is not the first's frames and then the second's"
        assert held["frames"] == 12 and held["every_frame_decodes_equal_to_its_source"] is True and [p["frames"] for p in held["from"]] == [7, 5], held
        assert not list((tmp / "out").glob("*.parts.txt")), "the list ffmpeg read was left behind"
        refused(lambda: tsl.lay_end_to_end([(tmp / "one.mkv", 7), (tmp / "two.mkv", 5)], tmp / "out" / "joined.mkv"), "exists", "new name")
        refused(lambda: tsl.lay_end_to_end([(tmp / "one.mkv", 6), (tmp / "two.mkv", 5)], tmp / "short.mkv"), "holds 7 frames where 6 are meant")
        assert not (tmp / "short.mkv").exists()
        refused(lambda: tsl.lay_end_to_end([(tmp / "one.mkv", 7), (tmp / "none.mkv", 5)], tmp / "none_out.mkv"), "not a file")
        write_mask_video(tmp / "wide.mkv", np.zeros((5, 24, 48), bool))
        refused(lambda: tsl.lay_end_to_end([(tmp / "one.mkv", 7), (tmp / "wide.mkv", 5)], tmp / "sizes.mkv"), "sizes or pixel formats")
        # a joined file that does not decode equal to its parts is removed and refused: here the reader is made to see one frame of it differ
        real = tsl.frame_hashes
        tsl.frame_hashes = lambda path: real(path)[:-1] + ["a frame that is not the source's"] if Path(path).name == "bad.mkv" else real(path)
        try:
            refused(lambda: tsl.lay_end_to_end([(tmp / "one.mkv", 7), (tmp / "two.mkv", 5)], tmp / "bad.mkv"), "did NOT decode equal", "removed")
        finally:
            tsl.frame_hashes = real
        assert not (tmp / "bad.mkv").exists(), "a joined file that did not decode equal was left where a job could read it"
        # a reference is another codec in and the lossless one out: still frame for frame what its parts decode to
        a_reference(tmp / "r1.mp4", 0, 7)
        a_reference(tmp / "r2.mp4", 50, 5)
        tsl.lay_end_to_end([(tmp / "r1.mp4", 7), (tmp / "r2.mp4", 5)], tmp / "ref.mkv")
        assert tsl.frame_hashes(tmp / "ref.mkv") == tsl.frame_hashes(tmp / "r1.mp4") + tsl.frame_hashes(tmp / "r2.mp4")
    return "12 frames, the first's seven and then the second's five, read back equal"


def tables_laid_end_to_end() -> str:
    first, second = a_table(40, 9), a_table(50, 11)
    said = {"what": "joined", "by": "a check"}
    joined = tsl.joined_table(first, second, said)
    assert joined["frames"] == 90 and joined["cuts"] == [40], (joined["frames"], joined["cuts"])
    assert joined["shots"][0] == first["shots"][0], "the first table's shot was changed"
    late = joined["shots"][1]
    # by hand, every field the tracker numbers by frame, moved on by the first table's 40 frames; and what is not a frame, as it was
    want = {"shot": 2, "first_frame": 40, "last_frame": 89, "frames": 50, "shown_frame": 51, "subject": {"state": "picked", "person": 1, "seed_frame": 51},
            "seeded": {"frame": 51, "box": [1.0, 2.0, 3.0, 4.0], "person": 1}, "frames_with_subject": 48, "frames_without_subject": [[43, 44]],
            "regained": [{"lost_from_frame": 43, "seed_frame": 45, "detections": 2}], "probes_after_a_loss": [{"frame": 44, "detections": 1}],
            "gallery_frames": [51], "looks": [{"frame": 46, "people": [{"person": 1, "box": [1, 2, 3, 4]}]}], "frames_on_others": [[47, 48]],
            "people": [{"person": 1, "box": [1, 2, 3, 4], "has_head": True}], "track_score": [0.5] * 50}
    assert late == want, {k: (late.get(k), v) for k, v in want.items() if late.get(k) != v}
    assert joined["gallery"]["frames"] == [9, 51] and len(joined["gallery"]["signatures"]) == 2 and joined["pick_frame"] == 9
    assert joined["joined_from"]["the_second_tables_own_in_its_own_frames"]["pick_frame"] == 11 and joined["joined_from"]["the_second_begins_on_table_frame"] == 40
    assert second["shots"][0]["first_frame"] == 0 and "joined_from" not in first, "a source table was changed in place"
    # the lane's own reader of a table cut to a stretch gets each source's shots back, at the source's own frame numbers
    for table, at, frames in ((first, 0, 40), (second, 40, 50)):
        cut = wiw.cut_to("a", joined, 100, 100 + at, frames)
        got = [(s["first_frame"], s["last_frame"], s["shown_frame"]) for s in cut["shots"]]
        assert got == [(s["first_frame"], s["last_frame"], s["shown_frame"]) for s in table["shots"]] and cut["cuts"] == [], (at, got, cut["cuts"])
    assert wiw.cut_to("a", joined, 100, 100, 90)["cuts"] == [40]
    # a table whose own shots hold a cut keeps it, moved on
    cutty = a_table(50, 11)
    cutty["cuts"] = [20]
    assert tsl.joined_table(first, cutty, said)["cuts"] == [40, 60]
    refused(lambda: tsl.joined_table(first, {**second, "size": [64, 48]}, said), "`size`")
    refused(lambda: tsl.joined_table(first, {**second, "version": 1}, said), "`version`")
    pose = tsl.joined_pose(a_pose(100, 40), a_pose(140, 50), said)
    assert [(r["frame"], r["source_frame"]) for r in pose["frames"]] == [(i, 100 + i) for i in range(90)] and pose["first_source_frame"] == 100
    refused(lambda: tsl.joined_pose(a_pose(100, 40), a_pose(141, 50), said), "do not meet")
    refused(lambda: tsl.joined_pose(a_pose(100, 40), {**a_pose(140, 50), "image_size": [64, 48]}, said), "`image_size`")
    return "a second table's frames moved on by the first's length, field by field; the cut where they meet"


def the_two_loads_are_held_to_each_other() -> str:
    with tempfile.TemporaryDirectory() as t:
        made = Made(Path(t))

        def with_b(name, change, *words):
            refused(lambda: made.job(b=str(made.changed(made.b, name, change))), *words)

        def with_a(name, change, *words):
            refused(lambda: made.job(a=str(made.changed(made.a, name, change))), *words)
        with_b("b_seed", lambda j: j["defaults"].update(seed=1), "`defaults`", "the window alone")
        # two loads of ONE seed asked for at two times (`shot_to_load.py job --seed N --seed-why`): the seed is the same and the words behind it
        # (`defaults.seed_by`) are a record, not a setting that differs; the two-shot load carries the first load's. THE CASE THAT GOES RED when
        # the records are held to each other as settings: a seed asked for twice could then never be made one load
        words = lambda at: {"why": f"asked at {at}", "by": "the session", "at": at}   # noqa: E731
        once = made.job(a=str(made.changed(made.a, "a_asked", lambda j: j["defaults"].update(seed=9, seed_by=words("T1")))),
                        b=str(made.changed(made.b, "b_asked", lambda j: j["defaults"].update(seed=9, seed_by=words("T2")))), name="asked_twice")
        assert "wrote" in once and json.loads((made.jobs / "asked_twice.json").read_text())["defaults"]["seed_by"] == words("T1"), once
        with_b("b_asked_other", lambda j: j["defaults"].update(seed=9, seed_by=words("T3")), "`defaults`", "the window alone")   # a's seed is not 9
        with_b("b_margin", lambda j: j["loads"][0].update(margin=64), "the load's settings")
        with_b("b_still", lambda j: j["job"]["subjects"][0].update(still="other.png"), "the subject a's keys")
        with_b("b_pieces", lambda j: j["job"]["subjects"][0].update(pieces="largest"), "the subject a's keys")
        with_b("b_form", lambda j: j["job"]["reference"].update(form="plain"), "the form of the reference")
        # a zoom a person asked for is refused; the chain's default zoom (every ordinary load's since 2026-10-14) is set aside (the_job_step)
        with_b("b_zoomed", lambda j: (j["loads"][0].update(render_region="part"), j["job"].update(zoomed={"why": "WHY", "said_by": "a person", "at": "t"})),
               "`render_region`", "not built")
        with_b("b_kept", lambda j: j["job"].update(keep_outs={"made": []}), "`keep_outs`", "not built")
        with_b("b_band", lambda j: j["loads"][0].update(hair_band=0.5), "`hair_band`", "not built")
        with_b("b_no_ref", lambda j: j["job"]["reference"].update(made=None), "no head reference made")
        with_b("b_other_subject", lambda j: (j["loads"][0].update(subject="b", others=["a"])), "two subjects")
        # the pick a tracker started from is the preview's business and may differ
        assert "wrote" in made.job(b=str(made.changed(made.b, "b_pick", lambda j: j["job"]["subjects"][0].update(pick_at=150))), name="picks_differ")
        # the two loads meet at the cut: the first ending on the frame before it, the second beginning on it
        with_a("a_short", lambda j: (j["job"].update(frames=30), j["loads"][0].update(frames=30)), "MEET AT THE CUT", "139")
        with_b("b_late", lambda j: (j["job"].update(first=142, frames=48), j["loads"][0].update(first=142, frames=48, motion_first=142)), "MEET AT THE CUT")
        refused(lambda: made.job(a=str(made.b), b=str(made.a)), "NEXT shot", "the first given first")
        refused(lambda: made.job(b=str(made.a)), "NEXT shot")
        # each preview is of its whole shot, so the two laid end to end have no hole
        pjob, folder = made.previews[SHOT_B[0]]
        (folder / "job.json").write_text(json.dumps({"job": {**pjob, "first": 139, "frames": 51}}))
        refused(lambda: made.job(), "of its whole shot", "no hole")
        # the two previews' trackers ran with different memory: their masks laid end to end are not one tracker's
        off = {**pjob, "sam_memory_selection": {"on": False, "why": "W", "said_by": "a person", "at": "t"}}
        (folder / "job.json").write_text(json.dumps({"job": off}))
        refused(lambda: made.job(), "different memory", "SAM 3.1 memory selection")
        assert not list((made.root / "data").glob("*a_two*")), "a refused two-shot load wrote its joined preview"
    return ("another seed, margin, still, form, subject; zoomed, kept out, banded; not meeting at the cut; out of order; the words behind one seed "
            "asked for at two times are not a difference")


def the_job_step() -> str:
    with tempfile.TemporaryDirectory() as t:
        made = Made(Path(t))
        refused(lambda: made.job(why=" "), "--why")
        made.m.dry = True
        said = made.job()
        assert "DRY: nothing written" in said and not (made.jobs / "a_two_0106.json").exists() and not list((made.root / "data").glob("*a_two*")), said
        assert not (made.m.output / "Video/v/the session/a_two_0106_joined").exists()
        made.m.dry = False
        said = made.job()
        d = json.loads((made.jobs / "a_two_0106.json").read_text())
        job, load = d["job"], d["loads"][0]
        # the job: both loads' frames, the record's cut, why and who, and the plainest load in everything else
        two = job["two_shots"]
        assert (two["cut"], two["shots"], two["said_by"]) == (CUT, [list(SHOT_A), list(SHOT_B)], "the session") and two["why"] == "one face across the cut, for one pair", two
        assert [x["name"] for x in two["loads"]] == ["a_fp_0106", "a_fp_0140"] and two["latent_steps_at_the_cut"]["the_cut_is"] == "on a latent step's edge"
        assert (job["first"], job["frames"], load["first"], load["frames"], load["motion_first"]) == (A_FIRST, 84, A_FIRST, 84, A_FIRST), (job["first"], job["frames"], load)
        ordinary = json.loads(made.a.read_text())
        assert d["defaults"] == ordinary["defaults"] and load["subject"] == "a" and load["others"] == ["b"] and load["text"] == "a_two_0106"
        assert job["texts"] == {"a_two_0106": {"file": "a_two_0106.txt", "frames": [A_FIRST, SHOT_B[1]], "by": ""}}, job["texts"]
        assert [x["first"] for x in job["left_out"]] == [100], "the frames the first load left as the source are no longer said"
        assert "NOT A DEFAULT" in job["note"] and "a_fp_0106" in job["note"] and "[Shot 2]" in job["note"]
        # THE CASE THAT GOES RED when the chain's default zoom reaches a two-shot load: the node refuses a cut inside a zoomed load
        assert ordinary["loads"][0].get("render_region") == "part" and ordinary["job"]["zoomed"]["said_by"] == "the chain's default", ordinary["job"].get("zoomed")
        assert "render_region" not in load and job["whole_frame"]["why"] == tsl.WHOLE_FRAME_WHY and "RENDERED ON THE WHOLE FRAME" in job["note"], (load, job.get("whole_frame"))
        graphs.check_job(job)
        graphs.check_load(job, masked_job.loads_of(d)[0], masked_job.chain_lengths())
        # the joined preview's folder says what it is, and every joined mask is its sources' frames, read back here
        pfolder = made.root / "data" / "2026-01-03_a_two_0106_joined_job"
        assert job["preview"] == str(pfolder), job["preview"]
        record = json.loads((pfolder / "preview.queued.json").read_text())
        assert record["status"] == "success" and "NOTHING WAS QUEUED" in record["status_is"] and [p["name"] for p in record["joined_from"]["previews"]] == ["shot_c_0100", "shot_c_0140"]
        joined = json.loads((pfolder / "job.json").read_text())["job"]
        assert "NOT A PREVIEW THAT RAN" in joined["note"]
        # THE CASE THAT GOES RED when the joined preview drops what its trackers ran with: a load on it would read "not recorded"
        memory = joined.get("sam_memory_selection") or {}
        assert memory.get("on") is True and memory.get("joined_from") == ["shot_c_0100", "shot_c_0140"], joined.get("sam_memory_selection")
        assert stl.memory_selection_carried(joined)[0].startswith("on:"), stl.memory_selection_carried(joined)
        out = made.m.output / "Video/v/the session/a_two_0106_joined"
        for label in "ab":
            for salt, kind in enumerate(tsl.KINDS):
                back = read_back(out / f"a_two_0106_joined_{label}_{kind}_00001.mkv")
                want = np.concatenate([pictures(SHOT_A[0], 40, salt + 10 * "abc".index(label)), pictures(SHOT_B[0], 50, salt + 10 * "abc".index(label))])
                assert back.shape == want.shape and (back == want).all(), f"the joined {kind} of {label} is not the two previews' frames end to end"
            table = json.loads((out / f"a_two_0106_joined_{label}_shots_00001_shots.json").read_text())
            assert table["frames"] == 90 and table["cuts"] == [40] and len(table["shots"]) == 2
            assert len(json.loads((out / f"a_two_0106_joined_{label}_pose_table.json").read_text())["frames"]) == 90
        # the reference: the two loads' own, end to end, on the load's frames
        ref = Path(load["motion_file"])
        assert ref.name == "a_head_reference_0106_0189_two_shots.mkv" and ref.parent.name == "a_0106_0189_two_shots", ref
        assert tsl.frame_hashes(ref) == [h for x in two["loads"] for h in tsl.frame_hashes(x["reference"])] and len(tsl.frame_hashes(ref)) == 84
        about = json.loads(ref.with_suffix(".json").read_text())
        assert about["frames"] == [A_FIRST, SHOT_B[1]] and about["frames_held"] == 84 and about["the_cut_is_on_source_frame"] == CUT, about
        assert job["reference"]["record"] == str(ref.with_suffix(".json"))
        assert job["reference"]["form"] == "drawn" and len(job["reference"]["laid_end_to_end_from"]) == 2
        # the chain reads it as any load: every mask from the joined preview's first frame, the capture handed the joined files
        ctx = lc.read_job(made.jobs / "a_two_0106.json", need_text=False)
        assert (ctx["preview"]["name"], ctx["preview"]["first"], ctx["preview"]["frames"], ctx["preview"]["order"]) == ("a_two_0106_joined", 100, 90, ["a", "b"]), ctx["preview"]
        specs = lc.mask_specs(ctx, lc.preview_files(ctx, made.m.output))
        assert len(specs) == 2 and all("at=100" in s and "shots_at=100" in s and "by=a_two_0106_joined" in s for s in specs), specs
        assert "pose_at=100" in specs[0] and "a_two_0106_joined_a_parts_00001.mkv" in specs[0], specs[0]
        (made.jobs / "a_two_0106.txt").write_text("[Shot 1] words. [Shot 2] The shot cuts to more words.\n")
        d["job"]["texts"]["a_two_0106"]["by"] = "a check"
        (made.jobs / "a_two_0106.json").write_text(json.dumps(d, indent=1))
        ctx = lc.read_job(made.jobs / "a_two_0106.json")
        ctx["job_folder"] = made.root / "data" / "2026-01-03_a_two_0106_job"
        ctx["capture"] = made.root / "data" / "2026-01-03_a_two_0106"
        (made.m.inputs / "a.png").write_bytes(b"")
        with contextlib.redirect_stdout(io.StringIO()):
            lc.step_first(ctx, made.m, argparse.Namespace(by="the session"))
        files = made.m.said("files first")
        assert files[files.index("--first") + 1] == str(A_FIRST) and files[files.index("--frames") + 1] == "84", files
        assert sum(1 for x in files if x == "--mask") == 2 and any("a_two_0106_joined_a_track_00001.mkv" in x for x in files)
        who = made.m.said("who is who")
        assert f"a={out / 'a_two_0106_joined_a_shots_00001_shots.json'}@100" in who and who[who.index("--frames") + 1] == "84", who
        # nothing is made twice under a name; the next letter is a load made again
        refused(lambda: made.job(), "exists")
        again = made.job(remake=True)
        assert "a_two_0106b" in again and (made.jobs / "a_two_0106b.json").is_file()
        # a load made again finds the reference of its subject and span there: read only because it still decodes equal to the two loads' own
        assert "the reference was there, and every frame of it decodes equal" in again, again
        assert json.loads((made.jobs / "a_two_0106b.json").read_text())["loads"][0]["motion_file"] == str(ref)
        a_reference(Path(two["loads"][1]["reference"]), 300, 50)             # the second load's reference is made again, with other pictures
        refused(lambda: made.job(remake=True), "does NOT decode equal", "another making")
        assert not (made.jobs / "a_two_0106c.json").exists()
        # two shots too long for one window are two loads: the ceiling made short for the case (one window holds a load
        # up to `MAX_LENGTH`, 2026-10-15), the chain under it
        real, real_max = masked_job.chain_lengths, graphs.MAX_LENGTH
        masked_job.chain_lengths, graphs.MAX_LENGTH = (lambda *_a, **_k: (39, 56)), 56
        try:
            refused(lambda: made.job(name="too_long"), "84 frames together", "two loads")
        finally:
            masked_job.chain_lengths, graphs.MAX_LENGTH = real, real_max
    return "the joined preview, the joined reference and the job file; read by the chain as any load"


def the_builder_takes_the_one_named_cut() -> str:
    with tempfile.TemporaryDirectory() as t:
        made = Made(Path(t))
        made.job()
        d = json.loads((made.jobs / "a_two_0106.json").read_text())
        job, one = d["job"], masked_job.loads_of(d)[0]
        # a capture as the builder reads one for its cuts: the manifest, and the JOINED table the tool wrote
        capture = made.root / "data" / "2026-01-03_a_two_0106"
        (capture / "subjects" / "a").mkdir(parents=True)
        (capture / "manifest.json").write_text(json.dumps({"first_frame": A_FIRST, "frames": 84, "subjects": [
            {"label": "a", "sightings": [{"by": "p", "shots_first_source_frame": 100}]}]}))
        joined = made.m.output / "Video/v/the session/a_two_0106_joined" / "a_two_0106_joined_a_shots_00001_shots.json"
        (capture / "subjects" / "a" / "shots__p.json").write_text(joined.read_text())
        assert masked_job.cuts_of(capture)[0] == [CUT], masked_job.cuts_of(capture)
        job["texts"]["a_two_0106"]["by"] = "a check"
        text = made.jobs / "a_two_0106.txt"

        def texts(words: str, **change):
            text.write_text(words)
            return masked_job.texts_for({**job, **change}, [one], made.jobs, capture)
        both = "[Shot 1] She stands in the hall. [Shot 2] The shot cuts to a closer view of her."
        assert texts(both)["a_two_0106"]["text"] == both
        # the same text in the same job without the key is a text of two shots, refused as it always was
        without = {k: v for k, v in job.items() if k != "two_shots"}
        text.write_text(both)
        refused(lambda: masked_job.texts_for(without, [one], made.jobs, capture), "cut(s) on 140", "a text is of one shot", "one load a shot", "`two_shots`")
        refused(lambda: texts("[Shot 1] She stands in the hall, and then closer."), "has no [Shot 2]", "each of its two shots under its own header")
        refused(lambda: texts("She stands. [Shot 2] The shot cuts."), "has no [Shot 1]")
        refused(lambda: texts(both + " [Shot 3] The shot cuts again."), "names a [Shot 3]")
        # the cut the job names is the cut the capture holds, and no other
        other = {**job["two_shots"], "cut": 150, "shots": [[100, 149], [150, 189]]}
        refused(lambda: texts(both, two_shots=other), "names the ONE cut on 150", "the cut(s) on 140", "no other")
        (capture / "subjects" / "a" / "shots__p.json").write_text(json.dumps({"shots": [{"first_frame": 0}]}))
        refused(lambda: texts(both), "names the ONE cut on 140", "no cut")
        refused(lambda: texts(both, two_shots={**job["two_shots"], "why": ""}), "`why`", "never a default")
    return "[Shot 1] and [Shot 2] across the named cut are taken; without the key, one header, a third, or another cut refused"


def the_latent_steps_at_the_cut() -> str:
    import capture_masked_run as cap
    loop_plan = cap._pack("loop_plan")
    cycle = loop_plan.STEP_CYCLE
    grid = tsl.step_grid(CUT, A_FIRST, 84, None)
    assert grid["the_cut_is"] == "on a latent step's edge" and grid["the_step_that_holds_it"] is None and not grid["the_captures_flags"], grid
    # every first frame the tool lists gives no split step by the planner's own function, and every other first frame of the cycle gives one
    assert grid["first_frames_that_put_the_cut_on_an_edge"][0] == A_FIRST, grid
    # a start one frame on puts the cut inside a step; the starts the tool then lists, and no others of the cycle, clear it
    late = A_FIRST + 1
    inside = tsl.step_grid(CUT, late, SHOT_B[1] - late + 1, None)
    listed = inside["first_frames_that_put_the_cut_on_an_edge"]
    assert listed and late not in listed and all(late <= f < late + cycle for f in listed), listed
    for first in range(late, late + cycle):
        split = loop_plan.split_steps([CUT], first, None, CUT - first + 20)
        assert (not split) == (first in listed), (first, split, listed)
    assert inside["the_cut_is"] == "INSIDE a latent step" and inside["the_step_that_holds_it"][0] < CUT <= inside["the_step_that_holds_it"][1], inside
    assert inside["frames_where_the_subject_is_on_both_sides_in_one_step"] and inside["the_captures_flags"] == ["region_shared_across_a_cut (iffy, blocks nothing)"], inside
    # the part on the first shot only: its region is carried onto the second shot's frames of that step, the flag that blocks
    carried = tsl.step_grid(CUT, late, SHOT_B[1] - late + 1, [[late, CUT - 1]])
    assert carried["frames_where_the_region_is_carried_onto_the_other_shot"] and all(f >= CUT for f in carried["frames_where_the_region_is_carried_onto_the_other_shot"]), carried
    assert carried["the_captures_flags"] == ["region_carried_across_a_cut (blocks the gate)"], carried
    # the part on neither side of the cut inside that step: nothing to fire on, though the cut is inside a step
    away = tsl.step_grid(CUT, late, SHOT_B[1] - late + 1, [[late, CUT - 10], [CUT + 10, 189]])
    assert away["the_cut_is"] == "INSIDE a latent step" and not away["the_captures_flags"], away
    assert any("neither" in line or "nothing of the cut" in line for line in tsl.grid_lines(away))
    # the two flags are the capture's, by the names and the levels its own source gives them
    source = (HERE / "capture_masked_run.py").read_text()
    assert '"rule": "region_carried_across_a_cut", "level": LEVELS[2]' in source and '"rule": "region_shared_across_a_cut", "level": LEVELS[1]' in source
    # the frames a part is on, read from a mask video as the tool reads one
    with tempfile.TemporaryDirectory() as t:
        mask = pictures(100, 10, 0)
        mask[3:6] = False
        write_mask_video(Path(t) / "part.mkv", mask)
        assert tsl.part_runs(Path(t) / "part.mkv", 100, 101, 9) == [[101, 102], [106, 109]]
    return f"a cycle of {cycle} frames: the listed starts and no others clear the cut"


def the_nodes_cuts_against_the_jobs() -> str:
    with tempfile.TemporaryDirectory() as t:
        made = Made(Path(t))
        made.job()
        ctx = lc.read_job(made.jobs / "a_two_0106.json", need_text=False)
        assert tsl.node_cuts(ctx, {"source": {"cuts": [CUT - A_FIRST]}}) == ([CUT], [CUT], True)
        assert tsl.node_cuts(ctx, {"source": {"cuts": []}}) == ([], [CUT], False)
        assert tsl.node_cuts(ctx, {"source": {"cuts": [CUT - A_FIRST, 60]}})[2] is False and tsl.node_cuts(ctx, {"source": {"cuts": [33]}})[2] is False
        refused(lambda: tsl.node_cuts(lc.read_job(made.a, need_text=False), {"source": {"cuts": []}}), "names no `two_shots`")
        a = argparse.Namespace(job_file=str(made.jobs / "a_two_0106.json"))
        refused(lambda: tsl.step_plan_cuts(a, made.m), "no plan file")
        plan = lc.plan_windows(ctx, made.m.output) / "a_two_0106_plan.json"
        plan.parent.mkdir(parents=True)
        for cuts, code, words in (([CUT - A_FIRST], 0, "THE SAME"), ([], 1, "it found no cut"), ([20], 1, "NOT THE SAME")):
            plan.write_text(json.dumps({"source": {"cuts": cuts}, "windows": [{"split_steps": []}]}))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                assert tsl.step_plan_cuts(a, made.m) == code, (cuts, out.getvalue())
            assert words in out.getvalue(), out.getvalue()
    return "the same list exits 0; none found, or another, exits 1"


def a_piece_on_two_shots() -> str:
    with tempfile.TemporaryDirectory() as t:
        made = Made(Path(t))
        shots, plan = made.read()
        piece = "Video/v/the session/a_two_0106/a_two_0106/a_two_0106_00001.mp4"
        rows = [{"first": A_FIRST, "last": SHOT_B[1], "piece": piece, "subject": "a"}]
        renders = [{"file": piece, "name": "a_two_0106", "first": A_FIRST, "last": SHOT_B[1], "as_the_graph_says": {"replace": None}, "region_exact": "face"}]
        out = shot_coverage.coverage(shots, plan, rows, [{"subject": "a", "first": 100, "last": 105, "why": "the cut on a step's edge"}], renders)
        mine = {(r["first"], r["last"]): r for r in out["rows"] if r["subject"] == "a"}
        one, two = mine[SHOT_A], mine[SHOT_B]
        assert [(p["frames"], p["piece"], p["crosses_a_cut"]) for p in one["laid"]] == [([A_FIRST, SHOT_A[1]], piece, True)], one["laid"]
        assert [(p["frames"], p["piece"], p["crosses_a_cut"]) for p in two["laid"]] == [([SHOT_B[0], SHOT_B[1]], piece, True)], two["laid"]
        assert one["state"] == "partly, as planned" and two["state"] == "as planned", (one["state"], two["state"])
        assert not mine[(190, 259)]["laid"], "the piece was laid on a shot it does not reach"
        # the page's item is the whole load's
        made.job()
        (made.jobs / "a_two_0106.txt").write_text("[Shot 1] words. [Shot 2] The shot cuts.\n")
        ctx = lc.read_job(made.jobs / "a_two_0106.json", need_text=False)
        item = lc.page_item(ctx, {k: "words" for k in ("title_rest", "state_words", "what", "look", "prompt_note")}, "clip.mp4", A_FIRST, SHOT_B[1], "media/clip.mp4", ("keep",))
        assert (item["first"], item["frames"], item["id"]) == (str(A_FIRST), "84", "a_two_0106") and "[Shot 2]" in item["prompt"], item
    return "laid on both shots' rows, each cut to its own frames, and marked as crossing a cut"


case("videos laid end to end decode equal to their sources, and one that does not is removed", videos_laid_end_to_end)
case("shot tables and pose tables laid end to end", tables_laid_end_to_end)
case("the two loads are held to each other, and to the record's cut", the_two_loads_are_held_to_each_other)
case("job: the joined preview, the joined reference, the job file; the chain reads it as any load", the_job_step)
case("the builder takes the one named cut under [Shot 1] and [Shot 2], and nothing else", the_builder_takes_the_one_named_cut)
case("the latent steps at the cut are the planner's own, and the flags the capture's", the_latent_steps_at_the_cut)
case("plan-cuts: the node's own cuts against the job's", the_nodes_cuts_against_the_jobs)
case("a piece on two record shots: both rows of the coverage table, one item on the page", a_piece_on_two_shots)

if __name__ == "__main__":
    sys.exit(finish())
