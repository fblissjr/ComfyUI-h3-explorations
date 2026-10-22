#!/usr/bin/env python3
"""Two neighbouring shots as ONE load in ONE window, made from the two shots' own ordinary loads. Not a default.

    <python> bench/two_shot_load.py job --a A.json --b B.json --shots SHOTS.json --folder INPUTS --by SESSION \\
        --why "why these two shots share a window" [--name NAME] [--remake] [--day DAY] [--dry]
        (a person writes the load's text, with a [Shot 1] and a [Shot 2]; `bench/shot_to_load.py text` says who;
        then `bench/load_chain.py first JOB.json`, and every step of the chain as for any load)
    <python> bench/two_shot_load.py plan-cuts JOB.json
        (after the chain's `plan`: the cuts the node found in the load, against the one the job names)

**What it is for** (the owner, 2026-10-13). Every load of the masked lane is one shot, and that stays the default:
a join at a cut costs nothing, and a long window costs more than two short ones. Two things one window for two
shots could buy, in the owner's words: one face for a subject across a cut, and a tough shot helped by an easier
one beside it. So this is one more tool, reached for on a person's word for one pair, and judged beside the same
pair made the ordinary way.

**What it is made from: the two ordinary loads.** `--a` and `--b` are the job files of two loads made the
ordinary way (`bench/shot_to_load.py`), one on each shot, with their previews and their head references made.
Nothing is tracked again and nothing is posed again. The two-shot load is those two laid end to end:

- **The masks.** Each preview's track, part and class-map videos of every subject are laid end to end, the first
  shot's frames and then the second's, into files of a new name, lossless, and EVERY FRAME OF EVERY JOINED FILE
  IS HELD EQUAL TO ITS SOURCE'S by its decoded hash, as its own planes and as RGB (`frame_hashes`), before
  anything is built on it. So the two-shot load's masks
  are the ordinary loads' own, and a pair made both ways differs in the window and not in a mask.
- **The shot tables and the pose tables** are joined the same way, the second's frame numbers moved on by the
  first's length (`joined_table`, `joined_pose`); a joined shot table holds one more cut than its two sources,
  where they meet.
- They go into a folder shaped as a preview's job folder (`data/<day>_<name>_job`, with `job.json` and
  `preview.queued.json`), SAID TO BE JOINED in both files: nothing was queued for it, and its record names the
  two previews it was laid from. `bench/load_chain.py` reads it as it reads any preview, by the name the job
  file gives it (`job.preview`), so the chain needs no rule of its own for a two-shot load.
- **The reference.** The two loads' head references laid end to end, lossless, held to their sources the same
  way. Each side keeps its own box and scale: the picture jumps at the cut, as the source does. It is filed by
  the subject and the span, as an ordinary reference is, so a two-shot load made again on the same frames finds
  it there, and reads it only while every frame of it still decodes equal to the two loads' references.
- **The job file**: one load on both loads' frames, with `two_shots` in its job block (the cut, the two shots,
  the two loads it was made from, why and who said so; `workflows/masked_job_graph.py::two_shots_of`). That key
  is the one thing by which the builder lets a load hold a cut (`bench/masked_job.py::held_as_two_shots`).
  WHO HOLDS THE CUT TO WHAT: this tool holds it to the shots record (`shots_of`). The builder does not read the
  record: it holds the key to itself (two neighbouring shots that meet on the cut) and to the capture's own shot
  tables. So a job file written by hand can name a cut the record does not have; the key is a person's word.

**What the two loads must be, each refused by name otherwise.** The same subject, source, canvas, still,
defaults and settings (seed, margin, feather, edge, the subject's options for the capture), and the same form
of reference: a pair made both ways is to differ in the window alone. On two NEIGHBOURING shots of the shots
record, the first load ending on the frame before the cut and the second beginning on it: the two-shot load
then replaces exactly the frames the two ordinary loads replace. Each preview of its whole shot, both tracking
the same subjects. Both shots in ONE window (`masked_job_graph.one_window`). NOT BUILT, and refused: loads that
do not meet at the cut, a subject one preview tracks and the other does not, a load of two subjects, a zoomed
load, a continued one, a hair band, keep-outs.

**What it prints before anything is built on the job: the latent steps at the cut.** A latent step is several
frames and its grid counts from the load's first frame (`docs/wiki/masked_v2v.md`, "A latent step is several
frames"), so whether the cut falls inside a step is arithmetic, done here with the planner's own function
(`loop_plan.split_steps`) on the frames the subject's part is on as the previews saved it: the step that holds
the cut, the frames on each side that would share it, which of the capture's two flags that is
(`region_carried_across_a_cut`, which blocks; `region_shared_across_a_cut`, which does not), and the first
frames, from the load's own and one cycle of steps on, that would put the cut on a step's edge. Data, with what it was read from: the capture reads the part its
own way (its pieces, its floor, its fill), and the gate is the capture's.

**`plan-cuts`.** The builder wires no shot table to the node, so the node finds a load's cuts itself, from the
picture (`video_mask.source_cuts`), and gates the composite by them (`cut_gate`). This reads the node's plan of
the load and prints its cuts beside the job's one. It exits 1 when they are not the same list: a node that
missed the cut gates nothing at it, and one that found another gates a frame nobody meant.

**What it cannot see.** Whether either ordinary load's masks are on the right person (the looks of those loads,
and of this one). Whether the text's two shot headers are true of their frames. What the model draws at a cut:
that takes the render, and the pair beside it.

Run it from ComfyUI's environment, with `H3_OUTPUT_DIR` and `H3_INPUT_DIR` set as the chain needs them.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "workflows"))
sys.path.insert(0, str(REPO))

import load_chain  # noqa: E402
from load_chain import Refused  # noqa: E402

#: The mask videos of a preview that are laid end to end, for every subject.
KINDS = ("track", "parts", "classes")
#: A load's keys that are its own place and words, and so differ between the two ordinary loads; every other key
#: of the two loads must be equal, and is carried into the two-shot load.
OWN_KEYS = ("name", "first", "frames", "text", "motion_file", "motion_first", "others")
#: The file `defaults` keys that are a record, not a setting, so they may differ between the two loads: `seed_by` is the words behind a
#: `--seed` of `shot_to_load.py job` (who, when), which two loads of one seed made at two times never share. `seed` itself must be the
#: same, and the two-shot load carries the first load's `defaults` whole, `seed_by` with them.
DEFAULTS_MAY_DIFFER = ("seed_by",)
#: A subject's keys in a job file that say where a tracker starts. A load tracks nobody, so they may differ.
TRACKER_KEYS = ("pick", "pick_at", "seeds", "phrase")
#: What of an ordinary load this tool does not carry into a two-shot load, as (where it is, the key, what it is).
NOT_BUILT = (("load", "subjects", "a load of two subjects"), ("load", "render_region", "a load rendered zoomed in"),
             ("load", "continue_from", "a load that continues another run"), ("load", "hair_band", "a hair band"),
             ("load", "audio", "an audio file in place of the source's"), ("load", "parts_mask", "a part mask of the load's own"),
             ("job", "keep_outs", "keep-outs"), ("job", "zoomed", "a load rendered zoomed in"),
             ("job", "who_is_who", "a who-is-who file handed in"), ("job", "two_shots", "a load that is itself of two shots"))
#: Why a two-shot load is rendered on the whole frame when its two loads were zoomed in by the chain's default alone (the
#: default of 2026-10-14, `shot_to_load.py`'s job step): the node refuses a cut inside a zoomed load (`video_mask`), and a
#: two-shot load holds one. A zoom a person asked for (`--zoomed`) is refused as before.
WHOLE_FRAME_WHY = ("a load of two shots holds a cut, and the node renders a zoomed load only inside one shot: the chain's default zoom of the two "
                   "loads is set aside, and this load is rendered on the whole frame")
#: Where a joined reference goes: by the video, the subject and the span, as an ordinary one does, and said to be two shots.
REFERENCES = "Video/{video}/references/{subject}_{first:04d}_{last:04d}_two_shots"
#: The keys of a shot table under which every whole number is a frame of the table, at any depth below them.
#: Read from `shot_table.py::build`; `bench/check_two_shot_load.py` holds a table of every one of them by hand.
FRAME_KEYS = ("first_frame", "last_frame", "shown_frame", "seed_frame", "frame", "lost_from_frame", "frames_without_subject",
              "gallery_frames", "frames_on_others", "cuts", "pick_frame")
#: What two tables or two pose tables of one job must agree on to be laid end to end.
TABLE_SAME, POSE_SAME = ("table", "version", "size"), ("schema", "image_size")


# ---------------------------------------------------------------- videos laid end to end

def frame_hashes(path) -> list[str]:
    """The hash of each decoded frame of a video, in order (ffmpeg's `framemd5`), taken TWICE and joined: of the
    frame's own planes, and of the frame as ffmpeg makes it RGB.

    Two files can hold the same planes and still be two pictures to whatever loads them, when one says its levels
    are full range and the other does not: a video copied into another container can lose that word. The planes'
    hash cannot see it and the RGB one can, so a joined file is held to both."""
    both = []
    for more in ([], ["-pix_fmt", "rgb24"]):
        done = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", *more, "-f", "framemd5", "-"], capture_output=True, text=True)
        if done.returncode:
            raise Refused(f"ffmpeg could not read {path}: {done.stderr.strip()[-300:]}")
        both.append([line.split()[-1] for line in done.stdout.splitlines() if line.strip() and not line.startswith("#")])
    if len(both[0]) != len(both[1]):
        raise Refused(f"{path} decodes to {len(both[0])} frames as its own planes and {len(both[1])} as RGB")
    return [f"{planes}.{rgb}" for planes, rgb in zip(*both)]


def pixel_format(path) -> tuple[str, int, int]:
    """A video's pixel format and size, by ffprobe."""
    done = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt,width,height", "-of", "json", str(path)],
                          capture_output=True, text=True)
    streams = json.loads(done.stdout or "{}").get("streams") or []
    if done.returncode or not streams:
        raise Refused(f"ffprobe could not read {path}")
    return streams[0]["pix_fmt"], int(streams[0]["width"]), int(streams[0]["height"])


def lay_end_to_end(parts: list[tuple[Path, int]], out: Path) -> dict:
    """Write `out`: the frames of each video in `parts`, in order, lossless; returns what was held. `parts` is
    [(a video, the frames it must hold)].

    Refused, with nothing left at `out`: a part that does not hold the frames it is said to; parts of two sizes
    or two pixel formats; and a joined file whose frames do not decode EQUAL, one for one, to its parts' frames.
    That last is the whole of what this tool promises of a mask, so it is held on every file every time and not
    only by the check."""
    out = Path(out)
    if out.exists():
        raise Refused(f"{out} exists: a joined file takes a new name")
    hashes, formats = [], set()
    for path, frames in parts:
        if not Path(path).is_file():
            raise Refused(f"{path} is not a file")
        mine = frame_hashes(path)
        if len(mine) != int(frames):
            raise Refused(f"{path} holds {len(mine)} frames where {frames} are meant: it is not the file of those frames")
        hashes += mine
        formats.add(pixel_format(path))
    if len(formats) != 1:
        raise Refused(f"the files to be laid end to end are of {len(formats)} sizes or pixel formats ({sorted(formats)}): not files of one job ({[str(p) for p, _ in parts]})")
    pix_fmt, width, height = next(iter(formats))
    out.parent.mkdir(parents=True, exist_ok=True)
    listing = out.with_name(out.name + ".parts.txt")
    listing.write_text("".join("file '" + str(Path(path).resolve()).replace("'", "'\\''") + "'\n" for path, _ in parts))
    try:
        done = subprocess.run(["nice", "-n", "19", "ffmpeg", "-v", "error", "-n", "-f", "concat", "-safe", "0", "-i", str(listing), "-map", "0:v:0",
                               "-c:v", "ffv1", "-level", "3", "-pix_fmt", pix_fmt, str(out)], capture_output=True, text=True)
    finally:
        listing.unlink(missing_ok=True)
    if done.returncode:
        out.unlink(missing_ok=True)
        raise Refused(f"ffmpeg could not write {out}: {done.stderr.strip()[-300:]}")
    made = frame_hashes(out)
    if made != hashes:
        out.unlink(missing_ok=True)
        differ = sum(1 for a, b in zip(made, hashes) if a != b)
        raise Refused(f"the joined file {out.name} did NOT decode equal to its parts ({len(made)} frames against {len(hashes)}, {differ} of those "
                      f"compared differ; pixel format {pix_fmt}): it is removed, and nothing is built on it")
    return {"file": str(out), "frames": len(made), "size": [width, height], "pixel_format": pix_fmt,
            "every_frame_decodes_equal_to_its_source": True, "from": [{"file": str(path), "frames": int(frames)} for path, frames in parts]}


# ---------------------------------------------------------------- tables laid end to end

def moved(value, by: int, under: bool = False):
    """`value` with every frame number in it moved on by `by`: the whole numbers under a key of `FRAME_KEYS`."""
    if isinstance(value, dict):
        return {k: moved(v, by, under or k in FRAME_KEYS) for k, v in value.items()}
    if isinstance(value, list):
        return [moved(v, by, under) for v in value]
    return value + by if under and type(value) is int else value


def joined_table(first: dict, second: dict, said: dict) -> dict:
    """Two trackers' shot tables of two neighbouring stretches as one: the first's shots, then the second's with
    every frame number moved on by the first's length and its shots numbered on from the first's.

    The cuts are the first's, the frame the two meet on, and the second's. What is one value a table (the pick,
    the match line) is the first's, and the second's is kept under `joined_from` with `said`, AS THE SECOND TABLE
    WROTE IT: its pick frame there is in the second table's own frames, the one frame number in the joined table
    that is not moved on. Which is why this table is nobody's tracker: a shot's number here is not a number a
    correction can be typed into."""
    for key in TABLE_SAME:
        if first.get(key) != second.get(key):
            raise Refused(f"the two shot tables differ in `{key}` ({first.get(key)!r} and {second.get(key)!r}): not tables of one job's form")
    n, count = int(first["frames"]), len(first["shots"])
    later: dict = moved({k: v for k, v in second.items() if k != "gallery"}, n)   # type: ignore[assignment]
    shots = list(first["shots"]) + [{**shot, "shot": count + i + 1} for i, shot in enumerate(later["shots"])]
    out = {**first, "frames": n + int(second["frames"]), "cuts": list(first["cuts"]) + [n] + list(later["cuts"]), "shots": shots,
           # the second table's one-a-table values AS IT WROTE THEM: its pick frame is in ITS OWN frames, not moved on, and the key says so
           "joined_from": {**said, "the_second_tables_own_in_its_own_frames": {k: second.get(k) for k in ("pick", "pick_frame", "pick_frame_named", "match", "match_named",
                                                                                                         "pick_seed", "match_floor", "match_moved_by_gap", "subject_phrase")},
                           "the_second_begins_on_table_frame": n,
                           "shots_are_numbered": f"the first table's 1 to {count} as they were; the second's {count + 1} on, its own number less {count}"}}
    if isinstance(first.get("gallery"), dict) and isinstance(second.get("gallery"), dict):
        a, b = first["gallery"], second["gallery"]
        out["gallery"] = {**a, **{k: list(a.get(k) or []) + list(b.get(k) or []) for k in ("signatures", "whole", "pick_probes")},
                          "frames": list(a.get("frames") or []) + [f + n for f in b.get("frames") or []],
                          "handed_in": (a.get("handed_in") or 0) + (b.get("handed_in") or 0)}
    if "track_score_trouble" in first or "track_score_trouble" in second:
        out["track_score_trouble"] = list(first.get("track_score_trouble") or []) + list(later.get("track_score_trouble") or [])
    return out


def joined_pose(first: dict, second: dict, said: dict) -> dict:
    """Two pose tables of two neighbouring stretches as one: the first's rows, then the second's with `frame` moved on."""
    for key in POSE_SAME:
        if first.get(key) != second.get(key):
            raise Refused(f"the two pose tables differ in `{key}` ({first.get(key)!r} and {second.get(key)!r}): not tables of one job's form")
    n = len(first["frames"])
    if first.get("first_source_frame") is None or second.get("first_source_frame") != first["first_source_frame"] + n:
        raise Refused(f"the first pose table holds {n} rows from source frame {first.get('first_source_frame')} and the second begins on "
                      f"{second.get('first_source_frame')}: they do not meet")
    return {**first, "frames": list(first["frames"]) + [{**row, "frame": int(row["frame"]) + n} for row in second["frames"]],
            "joined_from": {**said, "the_second_tables_own": {k: v for k, v in second.items() if k != "frames"}}}


# ---------------------------------------------------------------- the two loads, held to each other

def shots_of(shots: dict, a: dict, b: dict) -> tuple[dict, dict]:
    """The two NEIGHBOURING shots of the shots record that hold the two loads, the second load beginning on its
    shot's first frame: the cut is the record's."""
    rows = sorted(shots["shots"], key=lambda s: s["first"])
    span = lambda ctx: (ctx["load"]["first"], ctx["load"]["first"] + ctx["load"]["frames"] - 1)   # noqa: E731
    (a0, a1), (b0, b1) = span(a), span(b)
    at = next((i for i, s in enumerate(rows) if s["first"] <= a0 and a1 <= s["last"]), None)
    if at is None or at + 1 >= len(rows):
        raise Refused(f"the load {a['name']} ({a0}-{a1}) is inside no shot of the shots record that has a shot after it")
    one, two = rows[at], rows[at + 1]
    if not (two["first"] <= b0 and b1 <= two["last"]):
        raise Refused(f"the load {a['name']} is in the shot {one['first']}-{one['last']} and the load {b['name']} ({b0}-{b1}) is not inside the NEXT shot of the "
                      f"record, {two['first']}-{two['last']}: two neighbouring shots, the first given first")
    if one["last"] + 1 != two["first"]:
        raise Refused(f"the shots {one['first']}-{one['last']} and {two['first']}-{two['last']} of the record do not meet")
    if a1 + 1 != two["first"] or b0 != two["first"]:
        raise Refused(f"the load {a['name']} ends on {a1} and the load {b['name']} begins on {b0}, and the record's cut is on {two['first']}: the two loads "
                      f"must MEET AT THE CUT (the first ending on {two['first'] - 1}, the second beginning on {two['first']}), so that the one load replaces "
                      "exactly the frames the two replace. Loads that stop short of the cut are not built")
    return one, two


def zoomed_by_default(ctx: dict) -> bool:
    """A load zoomed in by the chain's default alone, not on a person's word (`job.zoomed.said_by`)."""
    return ctx["load"].get("render_region") == "part" and (ctx["job"].get("zoomed") or {}).get("said_by") == "the chain's default"


def held_to_each_other(a: dict, b: dict) -> None:
    """Refuse two ordinary loads that are not one thing made twice: a pair made both ways differs in the window alone.
    A zoom of the chain's default is no person's word and is set aside (`WHOLE_FRAME_WHY`), not refused."""
    for ctx in (a, b):
        for where, key, what in NOT_BUILT:
            if (where, key) in (("load", "render_region"), ("job", "zoomed")) and zoomed_by_default(ctx):
                continue
            if key in (ctx["load"] if where == "load" else ctx["job"]):
                raise Refused(f"the load {ctx['name']} has `{key}` ({what}): a two-shot load of that kind is not built")
    if a["subject"] != b["subject"]:
        raise Refused(f"the two loads are of two subjects ({a['subject']!r} and {b['subject']!r}): one subject across the cut")
    differs = []
    for what, x, y in (("the source", a["job"]["source"], b["job"]["source"]), ("the canvas", a["job"]["canvas"], b["job"]["canvas"]),
                       ("`defaults`", {k: v for k, v in a["defaults"].items() if k not in DEFAULTS_MAY_DIFFER},
                        {k: v for k, v in b["defaults"].items() if k not in DEFAULTS_MAY_DIFFER}),
                       ("the load's settings", {k: v for k, v in a["load"].items() if k not in OWN_KEYS}, {k: v for k, v in b["load"].items() if k not in OWN_KEYS}),
                       (f"the subject {a['subject']}'s keys", {k: v for k, v in a["subject_keys"].items() if k not in TRACKER_KEYS},
                        {k: v for k, v in b["subject_keys"].items() if k not in TRACKER_KEYS}),
                       ("the other subjects kept out", sorted(a["load"].get("others") or []), sorted(b["load"].get("others") or [])),
                       ("the subjects their previews track", sorted(a["preview"]["order"]), sorted(b["preview"]["order"])),
                       ("the form of the reference", (a["job"].get("reference") or {}).get("form"), (b["job"].get("reference") or {}).get("form"))):
        if x != y:
            differs.append(f"{what}: {json.dumps(x)} in {a['name']} and {json.dumps(y)} in {b['name']}")
    if differs:
        raise Refused(f"the two loads are not one thing made on two shots, in {len(differs)} thing(s):\n  - " + "\n  - ".join(differs)
                      + "\nA pair made both ways is to differ in the window alone, so the two ordinary loads agree in everything but their frames and their words. "
                        "(A subject one preview tracks and the other does not is not built.)")
    for ctx in (a, b):
        ref = ctx["job"].get("reference") or {}
        if not ref.get("made") or not ctx["load"].get("motion_file") or ctx["load"].get("motion_first") != ctx["load"]["first"]:
            raise Refused(f"the load {ctx['name']} has no head reference made on its own frames (`job.reference.made`, the load's `motion_file` beginning on its "
                          f"first frame): run `bench/shot_to_load.py reference` on it first")
        if (ctx["preview"]["first"], ctx["preview"]["frames"]) != (ctx["shot"]["first"], ctx["shot"]["last"] - ctx["shot"]["first"] + 1):
            raise Refused(f"the preview {ctx['preview']['name']} of the load {ctx['name']} is of {ctx['preview']['first']}-{ctx['preview']['first'] + ctx['preview']['frames'] - 1} "
                          f"and its shot is {ctx['shot']['first']}-{ctx['shot']['last']}: each preview is of its whole shot, so the two laid end to end have no hole")


def step_grid(cut: int, first: int, frames: int, present: list[list[int]] | None, loop_plan=None) -> dict:
    """The latent steps of a load at its cut, by the planner's own arithmetic (`loop_plan.split_steps`, `step_span`).

    `present` is the frames the subject's part is on, as [first, last] runs of source frames, or None for "on
    every frame". Returns the step that holds the cut when it is inside one, the frames of each kind
    (`across`: the region carried onto the other shot; `shared`: the subject on both sides in one step), the
    flag of the capture each is, and the first frames that put the cut on a step's edge."""
    if loop_plan is None:
        import capture_masked_run as cap
        loop_plan = cap._pack("loop_plan")
    at, n = loop_plan.step_span(cut - first)
    inside = first + at != cut
    split = loop_plan.split_steps([cut], first, present, frames)
    across, shared = sorted(f for s in split for f in s["across"]), sorted(f for s in split for f in s["shared"])
    return {"cut": cut, "load_first": first, "the_cut_is_load_frame": cut - first,
            "the_cut_is": "INSIDE a latent step" if inside else "on a latent step's edge",
            "the_step_that_holds_it": [first + at, first + at + n - 1] if inside else None,
            "frames_where_the_region_is_carried_onto_the_other_shot": across,
            "frames_where_the_subject_is_on_both_sides_in_one_step": shared,
            "the_captures_flags": (["region_carried_across_a_cut (blocks the gate)"] if across else []) + (["region_shared_across_a_cut (iffy, blocks nothing)"] if shared else []),
            # one cycle of first frames from the load's own: the grid repeats every cycle, and a later start drops frames of the first shot
            "first_frames_that_put_the_cut_on_an_edge": [f for f in range(first, min(first + loop_plan.STEP_CYCLE, cut)) if loop_plan.step_span(cut - f)[0] == cut - f],
            "read_from": "the frames the subject's part is on as the previews saved it" if present is not None else "the subject taken as on every frame: the worst case",
            "cannot_see": "the part as the capture reads it (its pieces, its floor, its fill), which is what the gate uses; and what the model draws in a step that holds two shots"}


def grid_lines(grid: dict) -> list[str]:
    lines = [f"THE LATENT STEPS AT THE CUT ({grid['read_from']}): the cut on {grid['cut']} is frame {grid['the_cut_is_load_frame']} of a load that begins on "
             f"{grid['load_first']}, and it is {grid['the_cut_is']}"
             + (f", the step of source frames {grid['the_step_that_holds_it'][0]}-{grid['the_step_that_holds_it'][1]}" if grid["the_step_that_holds_it"] else "") + "."]
    if grid["the_step_that_holds_it"]:
        for key, what in (("frames_where_the_region_is_carried_onto_the_other_shot", "the region is carried onto the OTHER shot's frames"),
                          ("frames_where_the_subject_is_on_both_sides_in_one_step", "the subject is on BOTH sides of the cut in that one step")):
            if grid[key]:
                lines.append(f"  {what}: {', '.join(map(str, grid[key]))}")
        lines.append("  the capture's gate will say: " + ("; ".join(grid["the_captures_flags"]) or "nothing of the cut: the part is on neither side of it in that step"))
        lines.append(f"  first frames that would put the cut on a step's edge, from this load's own and one cycle on: {', '.join(map(str, grid['first_frames_that_put_the_cut_on_an_edge']))}")
    else:
        lines.append("  no step holds frames of both shots, so neither of the capture's two cut flags has anything to fire on.")
    lines.append(f"  cannot see: {grid['cannot_see']}")
    return lines


def part_runs(path: Path, at: int, first: int, frames: int) -> list[list[int]]:
    """The frames of a load a mask video has anything on, as [first, last] runs of source frames. `at` is the source frame the file's frame 0 is on."""
    from _lib.frames import probe, stream
    w, h, _ = probe(str(path))
    on = [bool((frame > 127).any()) for frame in stream(str(path), (w, h), first - at, frames, vf=f"scale={w}:{h}:flags=neighbor")]
    runs: list[list[int]] = []
    for i, there in enumerate(on):
        if there and runs and runs[-1][1] == first + i - 1:
            runs[-1][1] = first + i
        elif there:
            runs.append([first + i, first + i])
    return runs


# ---------------------------------------------------------------- what is written

def joined_names(name: str, label: str) -> dict:
    """The file names of a joined preview's things for one subject, as a preview's own are named (`load_chain.preview_files` finds them by these)."""
    return {**{kind: f"{name}_{label}_{kind}_00001.mkv" for kind in KINDS}, "shots": f"{name}_{label}_shots_00001_shots.json", "pose": f"{name}_{label}_pose_table.json"}


def two_shot_job(a: dict, b: dict, name: str, preview_name: str, preview_folder: str, reference: dict, day: str, by: str, why: str, now: str,
                 output_key: str, grid: dict) -> dict:
    """The job file of the two-shot load, from the two ordinary loads' own: everything they share carried, their frames joined."""
    la, lb, ja = a["load"], b["load"], a["job"]
    first, last = la["first"], lb["first"] + lb["frames"] - 1
    cut, shots = lb["first"], [[a["shot"]["first"], a["shot"]["last"]], [b["shot"]["first"], b["shot"]["last"]]]
    load = {**{k: v for k, v in la.items() if k not in OWN_KEYS and k != "render_region"}, "name": name, "first": first, "frames": last - first + 1,
            "subject": a["subject"], "others": list(la.get("others") or []), "text": name, "motion_file": reference["file"], "motion_first": first}
    unzoomed = zoomed_by_default(a) or zoomed_by_default(b)
    made_from = [{"name": ctx["name"], "job_file": str(ctx["job_file"]), "first": ctx["load"]["first"], "frames": ctx["load"]["frames"],
                  "preview": ctx["preview"]["name"], "reference": ctx["load"]["motion_file"], "text": str(ctx["text"])} for ctx in (a, b)]
    note = (f"Written by bench/two_shot_load.py for {by}, {now}. TWO NEIGHBOURING SHOTS IN ONE LOAD, NOT A DEFAULT, on {by}'s word: {' '.join(why.split())}. "
            f"{a['subject']}, source frames {first} to {last} ({last - first + 1} frames): the shots {shots[0][0]}-{shots[0][1]} and {shots[1][0]}-{shots[1][1]} of the "
            f"record, the cut on {cut}. MADE FROM THE TWO ORDINARY LOADS {a['name']} ({la['first']}-{la['first'] + la['frames'] - 1}) and {b['name']} "
            f"({lb['first']}-{last}): their previews' masks laid end to end as {preview_name} (job.preview), every joined frame held equal to its source's; "
            f"their head references laid end to end, each side with its own box; their settings, which are equal. "
            f"THE TEXT IS NOT WRITTEN: {name}.txt beside this file is a person's, with a [Shot 1] and a [Shot 2], and texts.{name}.by says who wrote it. "
            f"THE CUT IS {grid['the_cut_is']} of the load ({'; '.join(grid['the_captures_flags']) or 'neither cut flag of the capture has anything to fire on'})."
            + (f" RENDERED ON THE WHOLE FRAME: {WHOLE_FRAME_WHY}." if unzoomed else ""))
    return {"job": {"name": name, "note": note, "source": ja["source"], "first": first, "frames": last - first + 1, "canvas": list(ja["canvas"]),
                    **({"source_frames": int(ja["source_frames"])} if ja.get("source_frames") is not None else {}),
                    "output": output_key,
                    "subjects": copy.deepcopy(ja["subjects"]), "left_out": list(ja.get("left_out") or []) + list(b["job"].get("left_out") or []),
                    "capture": f"data/{day}_{name}", "preview": preview_folder,
                    "texts": {name: {"file": f"{name}.txt", "frames": [first, last], "by": ""}},
                    **({"whole_frame": {"why": WHOLE_FRAME_WHY, "said_by": "bench/two_shot_load.py", "at": now}} if unzoomed else {}),
                    "two_shots": {"cut": cut, "shots": shots, "why": " ".join(why.split()), "said_by": by, "at": now, "loads": made_from, "latent_steps_at_the_cut": grid},
                    "reference": {"form": (ja.get("reference") or {}).get("form"), "why": (ja.get("reference") or {}).get("why"), "said_by": by, "made": now,
                                  "record": reference["record"], "laid_end_to_end_from": [ctx["job"]["reference"] for ctx in (a, b)]}},
            "defaults": copy.deepcopy(a["defaults"]), "loads": [load]}


def read_load(job_file: Path) -> dict:
    """An ordinary load as the chain reads it, with its file's `defaults`."""
    ctx = load_chain.read_job(Path(job_file), need_text=False)
    ctx["defaults"] = json.loads(ctx["raw"]).get("defaults") or {}
    return ctx


def write_reference(reference: dict, parts: list, loads: tuple, by: str, now: str, first: int, last: int, cut: int) -> None:
    """Lay the two loads' head references end to end at `reference["file"]`, held frame for frame, with its record beside it."""
    ref = lay_end_to_end(parts, Path(reference["file"]))
    Path(reference["record"]).write_text(json.dumps({
        "video": Path(reference["file"]).name, "what": "the head references of two neighbouring shots' loads laid end to end, for one load that holds both: "
        "each side is its own load's reference as it was made, with its own box and scale, so the picture jumps at the cut as the source does",
        "made_by": by, "made_at": now, "frames": [first, last], "its_first_frame_is_source_frame": first, "the_cut_is_on_source_frame": cut,
        "size": ref["size"], "frames_held": ref["frames"], "every_frame_decodes_equal_to_its_source": ref["every_frame_decodes_equal_to_its_source"], "pixel_format": ref["pixel_format"],
        "laid_end_to_end_from": [{"load": ctx["name"], "file": ctx["load"]["motion_file"], "frames": [ctx["load"]["first"], ctx["load"]["first"] + ctx["load"]["frames"] - 1],
                                  "its_own_record": (ctx["job"].get("reference") or {}).get("record")} for ctx in loads],
        "not_known": ["what a model makes of a reference whose framing jumps at the cut: the load's text says what the video is in each shot"]}, indent=1) + "\n")
    print(f"  the reference: {ref['frames']} frames, every one decoding equal to its source's: {reference['file']}")


# ---------------------------------------------------------------- the steps

def step_job(a, world) -> None:
    import masked_job
    import masked_job_graph as graphs
    import shot_coverage
    import shot_to_load as stl
    if not str(a.why or "").strip():
        raise Refused("--why takes a person's words: two shots share a window on somebody's word for this pair, and the word is kept in the job file")
    one, two = read_load(a.a), read_load(a.b)
    shots = shot_coverage.load_shots(a.shots)
    one["shot"], two["shot"] = shots_of(shots, one, two)
    held_to_each_other(one, two)
    # before anything is written: the trackers' memory, as each preview recorded it (`sam_memory_selection`, shot_to_load.preview_job): the joined masks
    # are both previews' frames, so the joined preview says what both ran with, and two that differ are refused
    ran = [json.loads((ctx["preview"]["folder"] / "job.json").read_text())["job"] for ctx in (one, two)]
    memory = [r.get("sam_memory_selection") for r in ran]
    if [(m or {}).get("on") for m in memory] != [(memory[0] or {}).get("on")] * 2:
        raise Refused(f"the previews {one['preview']['name']} and {two['preview']['name']} tracked with different memory (SAM 3.1 memory selection "
                      f"{[(m or {}).get('on', 'not recorded') for m in memory]}): their masks laid end to end would not be one tracker's. "
                      "Make the two previews alike, or the two-shot load is a person's word on the cut")
    first, cut = one["load"]["first"], two["load"]["first"]
    last = cut + two["load"]["frames"] - 1
    frames, windows = last - first + 1, masked_job.chain_lengths()
    if not graphs.one_window(frames, graphs.window_for(frames, windows), windows):
        raise Refused(f"the two loads are {frames} frames together and the longest one window is {max(windows)}: two shots too long for ONE window are two loads")
    subject, folder = one["subject"], Path(a.folder)
    video = load_chain.video_of(one, world.data)
    taken = lambda n: (folder / f"{n}.json").exists() or any(world.data.glob(f"????-??-??_{n}")) or any(world.data.glob(f"????-??-??_{n}_job"))   # noqa: E731
    name = stl.free_name(a.name or f"{subject}_two_{first:04d}", taken, a.remake)
    if not graphs.NAME.match(name):
        raise Refused(f"the name {name!r}: letters, digits and underscores only")
    preview_name = f"{name}_joined"
    output_key = stl.OUTPUT_KEY.format(video=video, session=a.by)
    joined_out = world.output / output_key / preview_name
    preview_folder = world.data / f"{a.day}_{preview_name}_job"
    ref_folder = world.output / REFERENCES.format(video=video, subject=subject, first=first, last=last)
    reference = {"file": str(ref_folder / f"{subject}_head_reference_{first:04d}_{last:04d}_two_shots.mkv")}
    reference["record"] = str(Path(reference["file"]).with_suffix(".json"))
    for path, what in ((joined_out, "folder of joined masks"), (preview_folder, "joined preview's job folder"), (folder / f"{name}.json", "job file")):
        if path.exists():
            raise Refused(f"the {what} {path} exists. Nothing is made twice under a name: --remake gives the load the next free letter")
    now, labels = load_chain.utc_now(), list(one["preview"]["order"])
    files = [load_chain.preview_files(ctx, world.output) for ctx in (one, two)]
    spans = [(ctx["preview"]["first"], ctx["preview"]["frames"]) for ctx in (one, two)]
    print(f"{name}: {subject}, source frames {first}-{last} ({frames} frames) in one window of {graphs.window_for(frames, windows)}: the shots "
          f"{one['shot']['first']}-{one['shot']['last']} and {two['shot']['first']}-{two['shot']['last']}, the cut on {cut}; from the loads {one['name']} and {two['name']}")
    # what is replaced: the part for a part load, the tracked subject for a whole one
    kind = "parts" if {**one["defaults"], **one["load"]}.get("carried", "track") == "held" else "track"
    grid = step_grid(cut, first, frames, None if world.dry else
                     part_runs(files[0][subject][kind], spans[0][0], first, cut - first) + part_runs(files[1][subject][kind], spans[1][0], cut, last - cut + 1))
    for line in grid_lines(grid):
        print(line)
    said = {"what": "laid end to end by bench/two_shot_load.py: NOT a tracker's own output and nothing queued", "by": a.by, "at": now,
            "previews": [{"name": ctx["preview"]["name"], "folder": str(ctx["preview"]["folder"]), "first": ctx["preview"]["first"], "frames": ctx["preview"]["frames"]}
                         for ctx in (one, two)]}
    job = two_shot_job(one, two, name, preview_name, _relative(preview_folder), reference, a.day, a.by, a.why, now, output_key, grid)
    stl.load_held_as_the_builder_holds(job, world)
    if world.dry:
        print(f"DRY: nothing written. It would lay {len(labels) * len(KINDS)} mask videos, {len(labels)} shot table(s) and the pose tables end to end into {joined_out}, "
              f"the two references into {reference['file']}, and write {preview_folder} and {folder / (name + '.json')}. The job was held to the builder's refusals "
              "that need neither its capture nor its text. NOT exercised: the joins themselves, and the latent steps were worked out with the subject on every frame.")
        return
    listed, held = [], {}
    for label in labels:
        names = joined_names(preview_name, label)
        for kind in KINDS:
            parts = [(f[label][kind], n) for f, (_at, n) in zip(files, spans)]
            if any(p is None for p, _ in parts):
                raise Refused(f"a preview saved no {kind} video for {label!r}: the two cannot be laid end to end")
            held[f"{label}.{kind}"] = lay_end_to_end(parts, joined_out / names[kind])
            listed.append(f"{output_key}/{preview_name}/{names[kind]}")
            print(f"  {label} {kind}: {held[f'{label}.{kind}']['frames']} frames, every one decoding equal to its source's")
        tables = [json.loads(Path(f[label]["shots"]).read_text()) for f in files]
        for table, (_at, n), ctx in zip(tables, spans, (one, two)):
            if int(table["frames"]) != n:
                raise Refused(f"{label}'s shot table of the preview {ctx['preview']['name']} holds {table['frames']} frames and the preview {n}")
        (joined_out / names["shots"]).write_text(json.dumps(joined_table(tables[0], tables[1], said)) + "\n")
        poses = [f[label]["pose"] for f in files]
        if all(poses):
            pose_a, pose_b = (json.loads(Path(p).read_text()) for p in poses)
            (joined_out / names["pose"]).write_text(json.dumps(joined_pose(pose_a, pose_b, said)) + "\n")
        elif any(poses) or label == subject and one["subject_keys"].get("pose", True) is not False:
            raise Refused(f"a pose table of {label!r} is in one preview and not the other, or the load's subject has none: the two cannot be laid end to end")
    ref_parts = [(ctx["load"]["motion_file"], ctx["load"]["frames"]) for ctx in (one, two)]
    if ref_folder.exists():
        # A reference belongs to a subject and a span, never to a load's name, so a two-shot load made again on the same
        # frames finds one there. It is not trusted because it is there: it is read only when every frame of it still
        # decodes equal to the two loads' references as they are now.
        if not Path(reference["file"]).is_file() or not Path(reference["record"]).is_file():
            raise Refused(f"{ref_folder} exists and does not hold both {Path(reference['file']).name} and its record: a folder a run left half-made is read and moved aside by a person")
        if frame_hashes(reference["file"]) != [h for path, _n in ref_parts for h in frame_hashes(path)]:
            raise Refused(f"{reference['file']} is there and does NOT decode equal to the two loads' references laid end to end now: a reference of another making. "
                          "It is read and moved aside by a person")
        print(f"  the reference was there, and every frame of it decodes equal to the two loads' references now: {reference['file']}")
    else:
        write_reference(reference, ref_parts, (one, two), a.by, now, first, last, cut)
    preview_folder.mkdir(parents=True)
    pjob = {"job": {"name": preview_name, "note": f"NOT A PREVIEW THAT RAN: the previews {one['preview']['name']} and {two['preview']['name']} laid end to end by "
                    f"bench/two_shot_load.py for {a.by}, {now}, for the two-shot load {name}. Source frames {spans[0][0]} to {spans[1][0] + spans[1][1] - 1}; every "
                    "joined mask frame was held equal to its source's.", "source": one["job"]["source"], "first": spans[0][0], "frames": spans[0][1] + spans[1][1],
                    "canvas": list(one["job"]["canvas"]), "output": output_key,
                    "subjects": ran[0]["subjects"], "left_out": [], "texts": {}, "joined_from": said,
                    **({"sam_memory_selection": dict(memory[0], joined_from=[one["preview"]["name"], two["preview"]["name"]])} if memory[0] else {})},
            "defaults": {}, "loads": []}
    (preview_folder / "job.json").write_text(json.dumps(pjob, indent=1))
    (preview_folder / "preview.queued.json").write_text(json.dumps({
        "stage": "preview", "status": "success", "prompt_id": None, "files": listed,
        "status_is": "the two previews' own, each of which ended in success: NOTHING WAS QUEUED for this folder", "joined_from": said, "held": held}, indent=1) + "\n")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(job, indent=1))
    print(f"wrote {preview_folder} and {folder / (name + '.json')}")
    print(f"IT STOPS HERE FOR THE TEXT: write {folder / (name + '.txt')}, one text with a [Shot 1] and a [Shot 2], from the two loads' own texts "
          f"({one['text']}, {two['text']}); then say who wrote it: bench/shot_to_load.py text {folder / (name + '.json')} --by {a.by} --by-hand \"why\" --writer \"who, and from which frames\". "
          f"Then bench/load_chain.py first, plan, `bench/two_shot_load.py plan-cuts`, and on as for any load.")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve())) if path.resolve().is_relative_to(REPO.resolve()) else str(path)


def node_cuts(ctx: dict, plan: dict) -> tuple[list[int], list[int], bool]:
    """(the cuts the node found, the cut the job names, whether they are the same list), all as source frames."""
    import masked_job_graph as graphs
    two = graphs.two_shots_of(ctx["job"])
    if two is None:
        raise Refused(f"{ctx['job_file']} names no `two_shots`: this step is for a load that holds a cut on its job's word")
    pad_before = int(((plan.get("windows") or [{}])[0]).get("pad_before", 0))
    n_load = int(ctx["load"]["frames"])
    found = [ctx["load"]["first"] + int(c) - pad_before for c in (plan.get("source") or {}).get("cuts") or []
             if not pad_before or (pad_before < int(c) < pad_before + n_load)]
    return found, [two["cut"]], found == [two["cut"]]


def step_plan_cuts(a, world) -> int:
    ctx = load_chain.read_job(Path(a.job_file), need_text=False)
    plan_file = load_chain.plan_windows(ctx, world.output) / f"{ctx['name']}_plan.json"
    if not plan_file.is_file():
        raise Refused(f"no plan file {plan_file}: the chain's `plan` step has not run to its end for this load")
    plan = json.loads(plan_file.read_text())
    found, named, same = node_cuts(ctx, plan)
    print(f"{ctx['name']}: the node's plan ({plan_file}) holds the cut(s) on source frame(s) {found or 'NONE'}; the job names the cut on {named[0]}.")
    split = [s for w in plan.get("windows") or [] for s in w.get("split_steps") or []]
    print(f"  the node's own split steps: {json.dumps(split) if split else 'none'}")
    if same:
        print("  THE SAME: the node gates the composite at the cut the job names.")
        return 0
    print("  NOT THE SAME. The node finds a load's cuts itself, from the picture (video_mask.source_cuts), since the builder wires it no shot table: "
          + ("it found no cut, so its cut gate does nothing at the job's cut." if not found else "it gates frames beside a cut the job does not name, or misses the job's.")
          + " This load is not rendered as it stands: it goes to the lead with this line.")
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = p.add_subparsers(dest="step", required=True)
    j = sub.add_parser("job", help="lay two ordinary loads' masks and references end to end and write the two-shot load's job file")
    j.add_argument("--a", required=True, help="the job file of the ordinary load on the FIRST shot")
    j.add_argument("--b", required=True, help="the job file of the ordinary load on the SECOND shot")
    j.add_argument("--shots", required=True, help="the video's shots record: the cut is the record's")
    j.add_argument("--folder", required=True, help="where the job file is written")
    j.add_argument("--by", required=True, help="the session")
    j.add_argument("--why", required=True, help="why these two shots share a window, in a person's words; kept in the job file")
    j.add_argument("--name", help="the load's name; <subject>_two_<first> when left out")
    j.add_argument("--remake", action="store_true", help="the next free letter after the name, for a load made again")
    j.add_argument("--day", default=datetime.date.today().isoformat(), help="the day in the names of the job's folders; today when left out")
    c = sub.add_parser("plan-cuts", help="after the chain's plan: the cuts the node found in the load, against the one the job names")
    c.add_argument("job_file")
    for s in sub.choices.values():
        s.add_argument("--host", default=load_chain.HOST)
        s.add_argument("--dry", action="store_true", help="job: print what would be made and write nothing")
    a = p.parse_args()
    import clip_folder
    try:
        if a.step == "plan-cuts":
            return step_plan_cuts(a, load_chain.World(a.host, a.dry))
        step_job(a, load_chain.World(a.host, a.dry))
    except (Refused, clip_folder.Refused) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
