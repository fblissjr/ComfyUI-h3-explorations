#!/usr/bin/env python3
"""After a cut, which named subject is each person on the frame? Several trackers' shot tables read together.

Each `MiniMaxH3SubjectTrack` decides a shot alone: every detection is compared with ONE reference, the subject on
the pick frame, and is the subject over a line (`subject_track.py::follow`). On 2026-10-10 a 29-frame shot of a load
with two named subjects was left empty by both trackers: each subject was there, each scored under its tracker's
line. This tool asks what the trackers' own data says when it is read for all subjects at once.

    galleries   no model. From K shot tables (each carries its subject's gallery, the signatures of up to eight
                frames of the picked shot's own track): how like its OWN gallery a subject's frame is, and how like
                each OTHER subject's, by the node's own gallery rule (`subject_track.gallery_scores`: per view the
                best over the gallery, then the lower of the views). Leave-one-out for the own gallery.
    looks       the model, outside the server. On named frames of the load every detection is compared with every
                subject's gallery, per view, and the frame is settled for all subjects together (below). Controls:
                the detections on a table's shown frame must be the table's people, and the top-third signature
                made here must be the node's own.
    corrections no model. From the K shot tables of one preview (one Subject Track per named subject, the same
                frames): every shot decided for all the subjects at once (`subject_tracks.hand_out`), over every
                look whose people carry signatures, and ONE FILE written for the gate: a row per shot with what
                each tracker called, what the sum says, its lead over the next way, and `ask` or `take`. It
                writes a correction as TEXT IN THAT FILE (`shot 2: person 4`, in the numbers of that tracker's
                own tile), never into a job. Everything is `ask` until a lead is measured
                (`subject_tracks.TOGETHER_LEAD`); `--unsure take` is the one switch that takes the top way anyway.
                A WEAK ROW SUGGESTS NOBODY (docs/wiki/who_is_who.md, section 6 and premortem 2: a yes to a weak
                suggestion is a guess with a person's name on it, and a typed answer is what later shots lean on).
                Under `--suggest-lead`, which is unset until a clip has measured its own, a differing row writes
                no correction: each tracker gets the numbered people of its own tile as `candidates` and an `ask`
                to show a person, who picks one or none. What the sum says stays in the file for tools.
                NOBODY HAS A VALUE (the first version of the design's "evidence total", docs/wiki/who_is_who.md
                section 6; 2026-10-10, after the tool agreed with a tracker that had taken the wrong person). A sum
                of likenesses is always positive, so somebody always beats nobody. A load's tables already hold
                looks KNOWN NOT to be a subject: every other subject's gallery, and whoever stood beside it on its
                own pick frame. The highest score any of those reaches against a subject is that subject's
                LEVEL on this clip: how like it a person who is not it can look. The evidence for a person is
                their score less the level, summed over the looks; nobody is worth zero, and any subject may
                take nobody. A row agrees only when the tracker, the sum and the evidence all say the same; where
                the evidence says nobody and a tracker or the sum has a person, the row differs and asks whether
                that person is the subject in a new look. The levels are in the file. They come from the picked
                shots alone, so a subject whose look has changed is under its own level too: that costs a
                question, never a take.
                ONLY COMPLETE LOOKS ARE READ (2026-10-10, found by the lead comparing two previews of one load,
                where the sum's answer for a subject followed whichever person the FIRST tracker did not hold).
                A tracker told who earlier trackers hold (`others`) lists a frame's people with those people
                removed, so its looks are not the frame: a person an earlier tracker holds is missing from them,
                is then "seldom seen", and the subject is handed whoever is left. A look is read only from a
                table that removed nobody there (`others_left_out`); the frames left unread are listed in the
                row, and a tracker whose own tile frame is not a complete look gets "cannot say", not a number.
                A SHOT A PERSON HAS ANSWERED IS NOT ASKED AGAIN: a tracker whose shot is picked, corrected by hand
                or seeded has a person's word (or a tool's) and is not this tool's to decide; the row says so and
                is settled by the trackers nobody has answered for, if any.
                A DETECTION WITH NO HEAD on any look it is seen on is listed (`people_seen`, `headless`) and never
                handed to anybody: a thing the detector called a person is not a candidate for a person.
    render      prints the tables of a `looks`, `galleries` or `corrections` JSON.

    <python> bench/who_is_who_across_shots.py galleries --table L=<shots.json> --table S=<shots.json> [--json J]
    <python> bench/who_is_who_across_shots.py looks --clip C --first-frame F --frames N --width W --height H \\
             --table L=<shots.json> --table S=<shots.json> --look 406,410,... --truth 'S@406-434=small' ... --json J [--cpu]
    <python> bench/who_is_who_across_shots.py corrections --table lead=<shots.json> --table second=<shots.json> --out J [--unsure take]
    <python> bench/who_is_who_across_shots.py render --json J

THREE VIEWS. The node signs a person in two places, the top third of the mask and the head. `looks` adds the WHOLE
mask. A table's gallery has no whole-mask signature, so for that view the gallery is made here: on the gallery's own
frames the detection most like the gallery entry is taken as the subject, and its whole mask is signed.

ONE SCALE (`--crops`, added 2026-10-10 after the first run). A signature is the trunk's features averaged under a mask
at the FRAME's scale, so a distant person is a handful of cells and a close one hundreds. With `--crops` every person
is also signed from a SQUARE CROP around their box (`CROP_MARGIN` wider than its longer side, black where it leaves
the frame), resized to the trunk's own input size, top third and whole mask; the galleries are made the same way on
their own frames. THE READING, WRITTEN BEFORE THAT RUN: signing from a crop is worth a place only if, on the middle
shot's looks, the small subject's lead over the other subject (and the lead of the right way of handing both out)
is wider than at the frame's scale on the frames where it was nothing, and no wider lead elsewhere is lost. One
shot of one clip; a wider lead here is a reason to try a second clip, not a default.

DUPLICATES. The detector can return one person several times, as nested masks. `merged` joins two detections when
most of the smaller mask lies inside the larger one (`SAME_PERSON`), and keeps the one the detector scored highest.

SETTLING A FRAME FOR ALL SUBJECTS (`settle`). The score of subject s on candidate c is the gallery rule's. A candidate
is s's when s and c are each other's best (no other candidate scores higher for s, no other subject scores higher on
c) and s leads every other subject on c by `--margin`. A subject with no such candidate has nobody on that frame.
Nothing is compared with a line.

ALL SUBJECTS AT ONCE, BY THE SUM (`together`), ADDED AFTER THE FIRST RUN and so not tested by it. `settle` asks each
subject for its best person, and a large, clear person is the best person of EVERY gallery (the first run: the lead's
person scored higher against the second subject's gallery than the second subject's own person did). `together`
gives every way of handing the frame's people to the subjects, one person to a subject at most, a total: the sum of
the scores it uses. The way with the highest total is the answer, and its lead over the next way is how sure it is.
With fewer people than subjects, a subject is left with nobody. `render` prints it per look and summed over a shot.

THE READING, WRITTEN BEFORE THE FIRST RUN (2026-10-10). Deciding all subjects together is worth building only if, on
every look of the two shots after the cut, `settle` gives the known answer (the shot where both are there: each
subject its own person; the shot where one is: that subject takes the person and the other gets NOBODY), with the
lead over the other subject at or above the margin on every take. A right answer on a smaller lead is luck and is
reported as that. The whole mask earns a place as a view only if it widens the smallest lead. What this cannot
show: a shot with a stranger and a named subject away, which this load does not have. One load, one clip.

Run from ComfyUI's environment, outside the server. `--cpu` keeps it off the card.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam3_parity_ladder import COMFY, REPO, boot, environment, load_core   # noqa: E402  (this folder's own tool)

sys.path.insert(0, str(REPO))

#: Two detections are one person when this share of the smaller mask lies inside the larger. Reasoned, not measured:
#: a person in front of another shares an edge with them, not most of their own area.
SAME_PERSON = 0.8
#: A loader start this many frames before the frame wanted, so that frame is the first it yields. Inherited: the
#: lead's builder for this load.
SEEK_BACK = 0.4
VIEWS = ("top third", "head", "whole")
#: The side of the square crop a person is signed from with `--crops`, as a multiple of their box's longer side.
#: Reasoned: room for hair and shoulders the mask's box cuts close, and little of anybody else.
CROP_MARGIN = 1.2


def read_tables(pairs: list[str], need_gallery: bool = True) -> dict[str, dict]:
    out = {}
    for pair in pairs:
        label, _, path = pair.partition("=")
        if not label or not path:
            raise SystemExit(f"--table wants LABEL=PATH; got {pair!r}")
        out[label] = json.loads(Path(path).read_text())
        if need_gallery and not (out[label].get("gallery") or {}).get("signatures"):
            raise SystemExit(f"{path}: this shot table carries no gallery")
    return out


def gallery(table: dict) -> list[list]:
    """A table's gallery: per frame, the signature of each view as an array, or None where that view was missing."""
    return [[None if v is None else np.asarray(v, dtype=np.float64) for v in views] for views in table["gallery"]["signatures"]]


def rule(views: list, held: list[list], use: tuple[int, ...]) -> float | None:
    """The node's gallery rule over the views in `use`: per view the best over the gallery, then the lowest. None when a view is missing."""
    worst = None
    for k in use:
        if k >= len(views) or views[k] is None:
            return None
        have = [g[k] for g in held if k < len(g) and g[k] is not None]
        if not have:
            return None
        best = max(float(g @ views[k]) for g in have)
        worst = best if worst is None else min(worst, best)
    return worst


def cmd_galleries(a):
    tables = read_tables(a.table)
    held = {label: gallery(t) for label, t in tables.items()}
    out = {"tables": {label: Path(pair.partition("=")[2]).name for label, pair in zip(tables, a.table)}, "rows": []}
    for label, mine in held.items():
        for i, views in enumerate(mine):
            row = {"subject": label, "gallery_frame": tables[label]["gallery"]["frames"][i], "has_head": views[1] is not None}
            for name, use in (("top third", (0,)), ("head", (1,)), ("rule", (0, 1))):
                row[name] = {"own": rule(views, mine[:i] + mine[i + 1:], use),
                             **{f"as {other}": rule(views, theirs, use) for other, theirs in held.items() if other != label}}
            out["rows"].append(row)
    out["line"] = _regain_line()
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    render_galleries(out)


def _regain_line() -> float:
    boot(float32=False, cpu=True)
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    sys.path.insert(0, str(COMFY))
    import _h3pack.subject_track as st
    return float(st.REGAIN_SAME)


def _f(v) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def render_galleries(out: dict):
    print("each gallery frame of a subject against the rest of its OWN gallery and against each OTHER subject's\n")
    print("| subject | gallery frame | head | view | own | as another | own minus the other |")
    print("|---|---|---|---|---|---|---|")
    over = total = 0
    leads = []
    for row in out["rows"]:
        for name in ("top third", "head", "rule"):
            cell = row[name]
            others = {k: v for k, v in cell.items() if k != "own"}
            best_other = max((v for v in others.values() if v is not None), default=None)
            lead = None if cell["own"] is None or best_other is None else cell["own"] - best_other
            print(f"| {row['subject']} | {row['gallery_frame']} | {'yes' if row['has_head'] else 'no'} | {name} | {_f(cell['own'])} "
                  f"| {', '.join(f'{k} {_f(v)}' for k, v in others.items())} | {_f(lead)} |")
            if name == "rule" and best_other is not None:
                total += 1
                over += best_other >= out["line"]
                leads.append(lead)
    print(f"\nby the rule (the lower of the two views): {over} of {total} frames score at or over the regain line "
          f"({out['line']}) against ANOTHER subject's gallery; own minus other is positive on "
          f"{sum(1 for v in leads if v is not None and v > 0)} of {len(leads)}, smallest {min(leads):.3f}")


# ----------------------------------------------------------------------------- every shot, for all subjects, from the tables

def _xyxy(box) -> tuple[float, float, float, float]:
    return (float(box[0]), float(box[1]), float(box[0] + box[2]), float(box[1] + box[3]))


def _best(sig, held: list) -> float | None:
    """A signature's best likeness over a gallery's; None when either is missing."""
    if sig is None or not held:
        return None
    v = np.asarray(sig, dtype=np.float64)
    return max(float(np.asarray(g, dtype=np.float64) @ v) for g in held)


AGREES, DIFFERS, UNRESOLVED = "agrees", "differs", "unresolved"


def hand_out_or_nobody(looks: list[dict[str, dict[int, float | None]]], levels: dict[str, float | None], seen_on_most: float) -> dict:
    """`subject_tracks.hand_out` with a zero: each score less its subject's level, and nobody (worth 0) open to every subject.

    `looks[i][subject][person]` as for `hand_out`; a subject whose level is None is left out (nothing known not to
    be it). People seen on fewer than `seen_on_most` of the looks are left out first, as there. Returns {"takes":
    {subject: person or None} or None when no look can be used, "total", "lead" over the next way, "next",
    "looks_used", "subjects"}.
    """
    import itertools
    subjects = sorted(s for look in looks for s in look if levels.get(s) is not None)
    subjects = sorted(set(subjects))
    everyone = sorted({p for look in looks for scores in look.values() for p in scores})
    seen = {p: sum(1 for look in looks if all(s in look and look[s].get(p) is not None for s in subjects)) for p in everyone}
    people = [p for p in everyone if looks and seen[p] >= seen_on_most * len(looks)]
    usable = [look for look in looks if all(s in look and all(look[s].get(p) is not None for p in people) for s in subjects)]
    out = {"takes": None, "total": None, "lead": None, "next": None, "looks_used": len(usable), "subjects": subjects}
    if not subjects or not usable:
        return out
    totals = {}
    for way in itertools.product(list(people) + [None], repeat=len(subjects)):
        taken = [p for p in way if p is not None]
        if len(taken) != len(set(taken)):
            continue
        totals[way] = sum(float(look[s][p]) - float(levels[s]) for look in usable for s, p in zip(subjects, way) if p is not None)
    # a tie goes to the way that hands out FEWER people, then to the one that reads first: nobody is the default
    order = sorted(totals, key=lambda w: (-round(totals[w], 9), sum(p is not None for p in w), tuple(-1 if p is None else p for p in w)))
    out.update(takes=dict(zip(subjects, order[0])), total=totals[order[0]])
    if len(order) > 1:
        out.update(lead=totals[order[0]] - totals[order[1]], next=dict(zip(subjects, order[1])))
    return out


def settle_tables(tables: dict[str, dict], least_lead: float | None = None, unsure: str = "ask", first_source_frame: int | None = None,
                  suggest_lead: float | None = None) -> dict:
    """Every shot of one load decided for all its trackers' subjects at once. See the module docstring, `corrections`.

    A person's score for a subject is their top third's best likeness over that subject's gallery plus their whole
    mask's (the two fail on different frames; one shot measured, 2026-10-10). A look is used when every person on it
    carries both signatures. `least_lead` None reads `subject_tracks.TOGETHER_LEAD`.

    THE NUMBER WRITTEN FOR A TRACKER IS READ ON THAT TRACKER'S OWN TILE FRAME, from its own list of people there
    (mrcorn's cold read, 2026-10-10: mapped from where a person stood on the LAST look, a subject who walked across
    the shot was written as the person she ended up beside). A tracker whose tile frame is not among the looks read
    gets no number: "cannot say", never a guess from another frame. A row that would hand two trackers one person
    on one frame is a fault of this tool and is written as unresolved, with no text to type.

    Each row's `status` is one of `agrees` (the sum would change nothing), `differs` (it would: `correction` is the
    text to type into that tracker) or `unresolved` (it cannot say). A row made from tables in which the correction
    is already typed agrees, so a second run after correcting clears itself. `verdict` is whether a differing row
    may be taken without a person (`subject_tracks.sure`), which today is always `ask`.

    `suggest_lead` is the lead at or over which a differing row is written with its correction filled in. None,
    the default, is "no clip has measured one": every differing row is then weak, `suggests` is False, `correction`
    is empty, and each tracker carries `candidates` (the people on its own tile, in tile order, each with this
    subject's total on them over the looks used) and `ask`, the question to put to a person. `would_type` keeps
    what the top way would have written, for a tool and for the status; it is not shown as a suggestion.
    """
    import subject_tracks as S
    labels = list(tables)
    first = tables[labels[0]]
    for label, t in tables.items():
        if t["frames"] != first["frames"] or list(t["cuts"]) != list(first["cuts"]) or list(t.get("size", [])) != list(first.get("size", [])):
            raise SystemExit(f"{label} and {labels[0]} are not tables of one load (frames, size or cuts differ): each tracker "
                             "must be given the same frames")
        if "whole" not in (t.get("gallery") or {}):
            raise SystemExit(f"{label}: this shot table was written before the tracker kept whole-mask signatures; run the preview again")
    # A TRACKER THAT PICKED NOBODY IS A FINDING, NOT A REASON TO STOP (the first three-tracker run, 2026-10-10: an
    # earlier tracker had wrongly taken a later subject on that subject's own pick frame, so `others` hid her from
    # her own tracker and she was picked nowhere; refusing the table hid exactly that). Such a subject has no
    # gallery, so nobody can be compared with it: the others are decided without it, and every row in which a
    # person is handed to somebody is unresolved, because that person could be the subject nobody picked.
    not_picked = {}
    for label, t in tables.items():
        if not [v for v in t["gallery"]["signatures"] if v and v[0] is not None] or not [v for v in t["gallery"]["whole"] if v is not None]:
            aside = sum(int((shot.get("others_left_out") or {}).get("on_frames_looked_at", 0)) for shot in t["shots"])
            not_picked[label] = (f"{label} has no gallery: its tracker picked nobody"
                                 + (f" (pick frame {t['pick_frame']})" if t.get("pick_frame") is not None else "")
                                 + (f"; {aside} detection(s) on the frames it looked at were set aside as another tracker's people, so its "
                                    "subject may be held by a tracker that runs before it" if aside else
                                    "; or its picked shot was settled by hand, which leaves no gallery"))
    labels = [label for label in tables if label not in not_picked]
    if not labels:
        raise SystemExit("no table here carries a gallery: no tracker picked anybody, so there is nobody to compare with. " + " ".join(not_picked.values()))
    everyone = list(tables)
    top = {label: [v[0] for v in tables[label]["gallery"]["signatures"] if v and v[0] is not None] for label in labels}
    whole = {label: [v for v in tables[label]["gallery"]["whole"] if v is not None] for label in labels}
    least = S.TOGETHER_LEAD if least_lead is None else least_lead

    def score_of(sig_top, sig_whole, label) -> float | None:
        a, b = _best(sig_top, top[label]), _best(sig_whole, whole[label])
        return None if a is None or b is None else a + b

    # Each subject's LEVEL: the highest score a look KNOWN NOT to be it reaches against its gallery. Known not to be
    # it: every other picked subject's gallery looks, and whoever stood beside it on its own pick frame.
    levels, levels_from = {}, {}
    for label in labels:
        known = []
        for other in labels:
            if other != label:
                held = tables[other]["gallery"]
                known += [(score_of(views[0], full, label), f"{other}'s gallery") for views, full in zip(held["signatures"], held["whole"])
                          if views and views[0] is not None and full is not None]
        picked = next((shot for shot in tables[label]["shots"] if shot["subject"]["state"] == "picked"), None)
        if picked is not None:
            known += [(score_of(p["signatures"][0], p["whole_signature"], label), f"a person beside {label} on its pick frame")
                      for p in picked["people"] if p["person"] != picked["subject"]["person"]
                      and (p.get("signatures") or [None])[0] is not None and p.get("whole_signature") is not None]
        known = [(v, src) for v, src in known if v is not None]
        levels[label] = max(known)[0] if known else None
        levels_from[label] = {"looks_known_not_to_be_it": len(known), "the_highest_is": max(known)[1] if known else None}
    rows = []
    for n in range(len(first["shots"])):
        by_frame: dict[int, list] = {}
        unread: set[int] = set()
        for label in everyone:
            shot = tables[label]["shots"][n]
            aside = shot.get("others_left_out") or {}
            for k, (frame, people) in enumerate([(shot["shown_frame"], shot["people"])] + [(look["frame"], look["people"]) for look in shot.get("looks", [])]):
                signed = [p for p in people if (p.get("signatures") or [None])[0] is not None and p.get("whole_signature") is not None]
                if not people or len(signed) != len(people):
                    continue
                # a list another tracker's people were removed from is not the frame: not read (see the module docstring)
                if int(aside.get("on_shown_frame", 0) if k == 0 else aside.get("on_frames_looked_at", 0)):
                    unread.add(int(frame))
                    continue
                if len(people) > len(by_frame.get(int(frame), [])):
                    by_frame[int(frame)] = people
        unread -= set(by_frame)
        frames = sorted(by_frame)
        names = S.same_people([[_xyxy(p["box"]) for p in by_frame[f]] for f in frames])
        # a detection with no head on ANY look it is seen on is a thing the detector called a person: listed, never handed out
        headed = {name for f, row in zip(frames, names) for name, p in zip(row, by_frame[f]) if len(p["signatures"]) > 1 and p["signatures"][1] is not None}
        headless = {name for row in names for name in row} - headed
        looks = []
        for f, row in zip(frames, names):
            looks.append({label: {name: score_of(p["signatures"][0], p["whole_signature"], label)
                                  for name, p in zip(row, by_frame[f]) if name not in headless} for label in labels})
        answer = S.hand_out(looks)
        evidence = hand_out_or_nobody(looks, levels, S.SEEN_ON_MOST)
        verdict, why = S.sure(answer, least)
        if verdict == S.ASK and unsure == "take" and answer["takes"] is not None:
            verdict, why = S.TAKE, f"UNSURE AND TAKEN BY THE SWITCH ({why})"
        on_frame = {f: {name: _xyxy(p["box"]) for name, p in zip(row, by_frame[f])} for f, row in zip(frames, names)}
        trackers, given = {}, []
        for label in labels:
            shot = tables[label]["shots"][n]
            called = {"state": shot["subject"]["state"], "person": shot["subject"]["person"]}
            tile = int(shot["shown_frame"])

            def on_tile(name, shot=shot, tile=tile):
                """A name as this tracker's own number for them on its own tile frame, or a sentence saying why not."""
                if name is None:
                    return None
                if tile not in on_frame:
                    return "cannot say: this tracker's tile frame is not among the looks read"
                if name not in on_frame[tile]:
                    return "cannot say: that person is not on this tracker's tile frame"
                # the same frame, so the same person's two boxes are one box: this tracker's own number for them
                overlap, person = max(((S.box_overlap(_xyxy(p["box"]), on_frame[tile][name]), p["person"]) for p in shot["people"]),
                                      default=(0.0, None))
                return person if overlap >= 0.9 else "cannot say: that person is not numbered on this tracker's tile"

            says, correction = "cannot say", ""
            if answer["takes"] is not None:
                says = on_tile(answer["takes"][label])
                if says is None:
                    correction = "" if called["person"] is None else f"shot {shot['shot']}: none"
                elif isinstance(says, int):
                    correction = "" if called["person"] == says else f"shot {shot['shot']}: person {says}"
                    given.append((label, tile, on_frame[tile][answer["takes"][label]]))
            # what the evidence says: the same reading, with nobody worth zero. A string when it cannot say
            if evidence["takes"] is None or label not in evidence["takes"]:
                proof = "cannot say: nothing in these tables is known not to be this subject" if levels.get(label) is None else "cannot say"
            else:
                proof = on_tile(evidence["takes"][label])
            # this subject's total, over the looks used, on each person of ITS OWN tile: what a person choosing is shown
            candidates = []
            if tile in on_frame:
                for p in shot["people"]:
                    overlap, name_there = max(((S.box_overlap(_xyxy(p["box"]), box), nm) for nm, box in on_frame[tile].items()), default=(0.0, None))
                    total = None
                    if overlap >= 0.9:
                        got = [look[label].get(name_there) for look in looks if label in look and look[label].get(name_there) is not None]
                        total = round(float(sum(got)), 3) if got else None
                    candidates.append({"person": p["person"], "box": p["box"], "total": total, "looks": 0 if total is None else len(got)})
            # the text to type exists only when the sum and the evidence say the same thing; otherwise it is a question
            agreed = isinstance(proof, str) or proof == says
            # a person's word (a pick, a typed correction) or a tool's (a seed) for this tracker in this shot
            word = ("picked" if called["state"] == "picked" else "corrected by hand" if shot.get("corrected") else
                    "seeded" if shot.get("seeded") else "")
            if word:
                correction = ""
            trackers[label] = {"called": called["state"], "called_person": called["person"], "the_sum_says": says,
                               "answered_by": word,
                               "the_evidence_says": proof, "correction": correction if agreed else "",
                               "would_type": correction if agreed else "", "candidates": candidates,
                               "typed_in_this_table": str(shot.get("corrected", "") or ""), "tile_frame": tile}
        fault = [(a[0], b[0]) for i, a in enumerate(given) for b in given[i + 1:] if a[1] == b[1] and S.box_overlap(a[2], b[2]) >= 0.9]
        cannot = answer["takes"] is None or any(isinstance(t["the_sum_says"], str) for t in trackers.values())
        if fault:
            for t in trackers.values():
                t["correction"] = t["would_type"] = ""
            cannot, why = True, f"A FAULT OF THIS TOOL: {fault[0][0]} and {fault[0][1]} were handed one person on one tile frame; nothing is written to type"
        for label in not_picked:
            shot = tables[label]["shots"][n]
            trackers[label] = {"called": shot["subject"]["state"], "called_person": shot["subject"]["person"], "answered_by": "",
                               "the_sum_says": "cannot say: this subject was never picked",
                               "the_evidence_says": "cannot say: this subject was never picked", "correction": "", "would_type": "", "candidates": [],
                               "typed_in_this_table": str(shot.get("corrected", "") or ""), "tile_frame": int(shot["shown_frame"])}
        # a row agrees only when the tracker, the sum and the evidence all say the same; the evidence disagreeing with
        # a take (it says nobody, a tracker or the sum has a person) is the wrong take this tool used to agree with
        # only the trackers nobody has answered for are decided: a pick, a typed correction and a seed are words, not calls
        open_ = {label: t for label, t in trackers.items() if label in labels and not t["answered_by"]}
        doubted = {label: t for label, t in open_.items() if not isinstance(t["the_evidence_says"], str)
                   and not isinstance(t["the_sum_says"], str) and t["the_evidence_says"] != t["the_sum_says"]}
        differs = any(t["would_type"] for t in open_.values()) or bool(doubted) or any(
            not isinstance(t["the_evidence_says"], str) and t["the_evidence_says"] != t["called_person"] for t in open_.values())
        cannot = cannot and (bool(fault) or any(isinstance(t["the_sum_says"], str) for t in open_.values()) or (answer["takes"] is None and bool(open_)))
        status = UNRESOLVED if cannot else DIFFERS if differs else AGREES
        note = ""
        if status == UNRESOLVED and not fault:
            # the reason a row is unresolved is what a person reads when it has no question to show (mrdeer, reading the
            # first real files: every row's `why` was the verdict's sentence about the lead, unresolved rows included)
            said = [f"{label}: {t['the_sum_says']}" for label, t in open_.items() if isinstance(t["the_sum_says"], str)]
            why = ("; ".join(said) if said and answer["takes"] is not None else
                   "no look on which every person can be compared with every subject")
        answered = {label: t["answered_by"] for label, t in trackers.items() if t["answered_by"]}
        if answered and not fault:
            note = ("answered, and not decided here: " + ", ".join(f"{label} ({how})" for label, how in answered.items())
                    + ("" if open_ else ". Nobody is left to decide in this shot"))
        if not_picked and not fault and answer["takes"] is not None and any(p is not None for p in answer["takes"].values()):
            status = UNRESOLVED
            why = (f"{', '.join(not_picked)} was never picked, so a person handed out in this shot could be them. "
                   "What the sum says of the others is kept for a reader, and nothing is suggested")
            for t in trackers.values():
                t["correction"] = ""
        # A shot in which no table lists anybody at all (a hand and an object; an empty room): everybody has nobody,
        # which is an answer and not a failure to compare. It agrees when every tracker called its subject absent.
        nobody_seen = not any(tables[label]["shots"][n]["people"] or any(look["people"] for look in tables[label]["shots"][n].get("looks", []))
                              for label in everyone)
        # ...and so is a shot whose only detections have no head on any look: things, listed and handed to nobody
        only_things = bool(frames) and not ({name for row in names for name in row} - headless)
        if (nobody_seen or only_things) and not fault:
            for label, t in trackers.items():
                t["the_sum_says"] = t["the_evidence_says"] = None
                t["correction"] = t["would_type"] = "" if t["called_person"] is None else f"shot {first['shots'][n]['shot']}: none"
            status = DIFFERS if any(t["would_type"] for t in trackers.values()) else AGREES
            note = ("no table lists anybody in this shot: every subject has nobody" if nobody_seen else
                    "every detection in this shot has no head on any look: listed as things, and every subject has nobody")
        # A shot every tracker PICKED its subject in is not decided here: a pick is a person's word (or the pick rule's,
        # which a person set), the pick frames are usually different frames showing different people, and a row that
        # blocks every job on its pick shot teaches people to override (mrdeer, reading the first real file,
        # 2026-10-10). What is still checked there: two trackers that picked ONE person on one frame.
        if not fault and all(t["called"] == "picked" for t in trackers.values()):
            picks = [(label, t["tile_frame"], next((_xyxy(p["box"]) for p in tables[label]["shots"][n]["people"] if p["person"] == t["called_person"]), None))
                     for label, t in trackers.items()]
            same = [(a[0], b[0]) for i, a in enumerate(picks) for b in picks[i + 1:]
                    if a[1] == b[1] and a[2] is not None and b[2] is not None and S.box_overlap(a[2], b[2]) >= 0.9]
            for t in trackers.values():
                t["would_type"] = t["correction"] = ""
            if same:
                status, why = UNRESOLVED, f"{same[0][0]} and {same[0][1]} picked one person on one frame: two trackers on one person"
            else:
                status, note = AGREES, "every tracker picked its subject in this shot; a pick is not decided here" + (
                    "" if answer["takes"] is not None else " (and its pick frames show different people, so nothing could be compared)")
        if unread and status != AGREES:
            note = (note + ". " if note else "") + (f"{len(unread)} look(s) not read, because another tracker's people had been removed from them: "
                                                   f"frames {', '.join(str(f) for f in sorted(unread)[:8])}" + (" and more" if len(unread) > 8 else ""))
        # a weak row suggests nobody: no text is filled in, and each tracker that would change gets a question
        suggests = (status == DIFFERS and not doubted
                    and (verdict == S.TAKE or (suggest_lead is not None and answer["lead"] is not None and answer["lead"] >= suggest_lead)))
        for label, t in trackers.items():
            t["ask"] = ""
            if status == DIFFERS and not suggests and not t.get("answered_by"):
                proof = t.get("the_evidence_says")
                wrong_for_evidence = label in labels and not isinstance(proof, str) and proof != t["called_person"]
                if t["would_type"] or label in doubted or wrong_for_evidence:
                    numbers = ", ".join(str(c["person"]) for c in t["candidates"]) or "nobody is numbered"
                    t["ask"] = (f"shot {first['shots'][n]['shot']}, {label}'s tile (frame {t['tile_frame']}): which numbered person is {label}? "
                                f"{numbers}, or none. Type `shot {first['shots'][n]['shot']}: person K` or `shot {first['shots'][n]['shot']}: none` into {label}'s tracker")
                    if proof is None and (t["called_person"] is not None or isinstance(t["the_sum_says"], int)):
                        who = t["called_person"] if t["called_person"] is not None else t["the_sum_says"]
                        t["ask"] += (f". The evidence says NOBODY: person {who} is no more like {label} than people known not to be {label} "
                                     f"on this clip are. If that is {label} in a new look, type the number; if not, type none")
                t["correction"] = ""
        # everybody seen in the shot, for a reader that must tell a person from a thing that never moves
        seen = {}
        for f, row in zip(frames, names):
            for name, p in zip(row, by_frame[f]):
                box = _xyxy(p["box"])
                seen.setdefault(name, []).append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, (box[2] - box[0]) * (box[3] - box[1])))
        people_seen = [{"name": name, "looks_seen": len(v), "headless": name in headless,
                        "centre_moved_px": round(max(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a in v for b in v), 1),
                        "box_area_px": [int(min(x[2] for x in v)), int(max(x[2] for x in v))],
                        "left_out_as_seldom_seen": name in answer["left_out"],
                        "handed_to": next((label for label in labels if answer["takes"] is not None and answer["takes"][label] == name), None)}
                       for name, v in sorted(seen.items())]
        rows.append({"shot": first["shots"][n]["shot"], "frames": [first["shots"][n]["first_frame"], first["shots"][n]["last_frame"]],
                     "status": status, "note": note, "suggests": bool(suggests), "verdict": verdict, "why": why,
                     "looks_with_signatures": frames, "looks_not_read_as_incomplete": sorted(unread),
                     "looks_used": answer["looks_used"], "people": answer["people"],
                     "lead": answer["lead"], "evidence_lead": evidence["lead"], "trackers": trackers, "people_seen": people_seen})
    return {"what": "every shot of one load decided for all its trackers' subjects at once; a file for the gate, not a job",
            "status_is": f"{AGREES}: the sum would change nothing. {DIFFERS}: it would, and `correction` is the text to type into that "
                         f"tracker. {UNRESOLVED}: it cannot say. A row agrees once its correction is typed and the preview run again",
            "frames_are": "the load's own frame numbers" + ("" if first_source_frame is None else f"; the load's first frame is the source's {int(first_source_frame)}, as given"),
            "first_source_frame": first_source_frame,
            "suggest_lead": suggest_lead,
            "suggest_is": "a differing row whose lead is under `suggest_lead` (or every differing row, when it is null: no clip has measured "
                          "one) writes no correction: `suggests` is false, each tracker has `candidates` and an `ask` to show a person. "
                          "`would_type` is what the top way would have written, for tools; do not show it as a suggestion",
            "not_picked": not_picked,
            "levels": levels, "levels_from": levels_from,
            "levels_are": "per subject, the highest score (top third plus whole mask, each the best over the subject's gallery) that a look "
                          "KNOWN NOT to be that subject reaches: another subject's gallery look, or a person beside it on its own pick frame. "
                          "The evidence for a person is their score less this, summed over the looks used; nobody is worth zero. From the "
                          "picked shots of this load only: a starting value for this clip, not a line",
            "tables": everyone, "least_lead": least, "unsure": unsure, "rows": rows}


def cmd_corrections(a):
    tables = read_tables(a.table, need_gallery=False)
    out = settle_tables(tables, unsure=a.unsure, first_source_frame=a.first_frame, suggest_lead=a.suggest_lead)
    out["files"] = {label: Path(pair.partition("=")[2]).name for label, pair in zip(tables, a.table)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    render_corrections(out)


def render_corrections(out: dict):
    labels = out["tables"]
    for label, why in (out.get("not_picked") or {}).items():
        print(f"NOT PICKED: {why}. The other subjects are decided without it, and no row that hands a person out is resolved.\n")
    if out.get("levels"):
        print("levels (how like each subject a person who is NOT it can look, on this load): "
              + "; ".join(f"{label} {_f(v)} ({out['levels_from'][label]['looks_known_not_to_be_it']} looks, the highest {out['levels_from'][label]['the_highest_is']})"
                          for label, v in out["levels"].items()) + "\n")
    print(f"{out['what']}\n{out['status_is']}\nleast lead to take without a person: {out['least_lead']}; unsure rows: {out['unsure']}; {out['frames_are']}\n")
    print("| shot | frames | status | looks used | people | lead | " + " | ".join(f"{label}: called, the sum says" for label in labels) + " | without a person | to type |")
    print("|---|---|---|---|---|---|" + "---|" * len(labels) + "---|---|")
    for row in out["rows"]:
        cells = []
        for label in labels:
            t = row["trackers"][label]
            called = t["called"] + ("" if t["called_person"] is None else f" (person {t['called_person']})")
            says = "nobody" if t["the_sum_says"] is None else (f"person {t['the_sum_says']}" if isinstance(t["the_sum_says"], int) else t["the_sum_says"])
            proof = t.get("the_evidence_says", "")
            proof = "nobody" if proof is None else (f"person {proof}" if isinstance(proof, int) else "cannot say" if proof else "")
            cells.append(f"{called}; {says}" + (f"; evidence: {proof}" if proof else ""))
        typed = "; ".join(f"{label}: `{row['trackers'][label]['correction']}`" for label in labels if row["trackers"][label]["correction"])
        asked = "; ".join(row["trackers"][label].get("ask", "") for label in labels if row["trackers"][label].get("ask"))
        typed = typed or (f"NOT SUGGESTED. {asked}" if asked else "nothing")
        print(f"| {row['shot']} | {row['frames'][0]}-{row['frames'][1]} | {row['status']} | {row['looks_used']} | {row['people']} | {_f(row['lead'])} | "
              + " | ".join(cells) + f" | {row['verdict']}: {row['why']}" + (f". {row['note']}" if row.get("note") else "") + f" | {typed} |")


# ----------------------------------------------------------------------------- the model


def merged(masks, scores, same: float = SAME_PERSON) -> tuple[list[int], dict[int, list[int]]]:
    """Which detections stand for a person once nested ones are joined: (kept indices, {kept: the ones it stands for})."""
    on = [(m > 0.5) for m in masks]
    area = [int(o.sum()) for o in on]
    order = sorted(range(len(on)), key=lambda i: -scores[i])
    kept, stands = [], {}
    for i in order:
        for k in kept:
            small = min(area[i], area[k])
            if small and int((on[i] & on[k]).sum()) >= same * small:
                stands[k].append(i)
                break
        else:
            kept.append(i)
            stands[i] = [i]
    return kept, stands


def settle(scores: dict[str, list[float | None]], margin: float) -> dict[str, dict]:
    """One frame for all subjects: `scores[subject][candidate]`. See the module docstring."""
    out = {}
    n = len(next(iter(scores.values()), []))
    for s, row in scores.items():
        able = [c for c in range(n) if row[c] is not None]
        if not able:
            out[s] = {"takes": None, "why": "nobody it can be compared with"}
            continue
        c = max(able, key=lambda k: float(row[k]))
        rivals = [float(scores[o][c]) for o in scores if o != s and scores[o][c] is not None]
        lead = float(row[c]) - max(rivals) if rivals else None
        if lead is not None and lead <= 0:
            out[s] = {"takes": None, "best": c, "score": row[c], "lead": lead, "why": "its best is more like another subject"}
        elif lead is not None and lead < margin:
            out[s] = {"takes": None, "best": c, "score": row[c], "lead": lead, "why": "its lead over another subject is under the margin"}
        else:
            out[s] = {"takes": c, "best": c, "score": row[c], "lead": lead, "why": "its best, and it leads" if rivals else "its best; no other subject can be compared"}
    return out


def together(scores: dict[str, list[float | None]]) -> dict | None:
    """Every subject at once: the handing-out of people to subjects with the highest total score, and its lead.

    `scores[subject][candidate]`. A subject may be left with nobody only when there are fewer people than subjects.
    None when some subject cannot be compared with a person it would have to take.
    """
    import itertools
    subjects = list(scores)
    n = len(next(iter(scores.values()), []))
    if not subjects or not n:
        return None
    slots = list(range(n)) + [None] * max(len(subjects) - n, 0)
    ways = {}
    for way in set(itertools.permutations(slots, len(subjects))):
        got = [scores[s][c] for s, c in zip(subjects, way) if c is not None]
        if any(v is None for v in got):
            continue
        ways[way] = float(sum(got))
    if not ways:
        return None
    order = sorted(ways, key=lambda w: -ways[w])
    return {"takes": dict(zip(subjects, order[0])), "total": ways[order[0]],
            "lead": None if len(order) < 2 else ways[order[0]] - ways[order[1]],
            "next": None if len(order) < 2 else dict(zip(subjects, order[1]))}


def cmd_looks(a):
    torch = boot(float32=False, cpu=a.cpu)
    import server
    server.PromptServer.instance = types.SimpleNamespace(prompt_queue=None, routes=None)
    sys.path.insert(0, str(COMFY / "custom_nodes" / "ComfyUI-VideoHelperSuite"))
    from videohelpersuite.load_video_nodes import LoadVideoFFmpegPath
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    import comfy.model_management as mm
    import comfy.utils

    import _h3pack.shot_table as shot_table
    import _h3pack.subject_track as st
    import sam31_corrections
    from h3_config import SEGMENTER, SUBJECT_TRACK

    torch.set_grad_enabled(False)
    tables = read_tables(a.table)
    held = {label: gallery(t) for label, t in tables.items()}
    looks = sorted({int(v) for v in a.look.split(",") if v.strip()})
    whole_frames = {label: t["gallery"]["frames"][:: max(1, int(a.gallery_every))] for label, t in tables.items()}
    start = round(max(0.0, (a.first_frame - SEEK_BACK) / a.rate), 6) if a.first_frame else 0.0
    video = LoadVideoFFmpegPath().load_video(video=a.clip, force_rate=0.0, custom_width=a.width, custom_height=a.height,
                                             frame_load_cap=int(a.frames), start_time=start, format="AnimateDiff")[0][..., :3]
    if int(video.shape[0]) != int(a.frames):
        raise SystemExit(f"the loader gave {int(video.shape[0])} frames, not {a.frames}")
    model, clip = load_core(SEGMENTER)
    if a.corrected:
        model, clip = sam31_corrections.corrected(model, clip)
    max_people = int(a.max_people or SUBJECT_TRACK["max_people"])
    detect, sign, _ = st._sam_callables(model, clip, video, SUBJECT_TRACK["subject_phrase"], SUBJECT_TRACK["detection_threshold"],
                                        max_people, SUBJECT_TRACK["head_phrase"])
    features = {}

    def whole(f: int, mask):
        """The whole mask's signature: the trunk as `subject_track._sam_callables` runs it, held against its own top third below."""
        if f not in features:
            mm.load_model_gpu(model)
            x = comfy.utils.common_upscale(video[f:f + 1, ..., :3].movedim(-1, 1), st.TRUNK_SIDE, st.TRUNK_SIDE, "bilinear",
                                           crop="disabled").to(device=mm.get_torch_device(), dtype=model.model.get_dtype())
            trunk = model.model.diffusion_model.detector.backbone["vision_backbone"].trunk(x)
            trunk = trunk[-1] if isinstance(trunk, (list, tuple)) else trunk
            features[f] = trunk[0].to(torch.float32).cpu()
        return st.signature(features[f], mask), st.signature(features[f], st.top_third(mask))

    def cropped(f: int, mask):
        """A person signed from a square crop around their box at the trunk's own size: (whole mask, top third), or (None, None)."""
        box = shot_table._box(mask)
        if box is None:
            return None, None
        x, y, w, h = box
        side = int(round(max(w, h) * CROP_MARGIN))
        left, top = int(round(x + w / 2 - side / 2)), int(round(y + h / 2 - side / 2))
        height, width = int(video.shape[1]), int(video.shape[2])
        canvas = torch.zeros((side, side, 3), dtype=video.dtype)
        held = torch.zeros((side, side), dtype=torch.float32)
        x0, y0, x1, y1 = max(left, 0), max(top, 0), min(left + side, width), min(top + side, height)
        canvas[y0 - top:y1 - top, x0 - left:x1 - left] = video[f, y0:y1, x0:x1, :3]
        held[y0 - top:y1 - top, x0 - left:x1 - left] = (mask[y0:y1, x0:x1] > 0.5).to(torch.float32)
        mm.load_model_gpu(model)
        pixels = comfy.utils.common_upscale(canvas[None].movedim(-1, 1), st.TRUNK_SIDE, st.TRUNK_SIDE, "bilinear",
                                            crop="disabled").to(device=mm.get_torch_device(), dtype=model.model.get_dtype())
        trunk = model.model.diffusion_model.detector.backbone["vision_backbone"].trunk(pixels)
        trunk = (trunk[-1] if isinstance(trunk, (list, tuple)) else trunk)[0].to(torch.float32).cpu()
        full, third = st.signature(trunk, held), st.signature(trunk, st.top_third(held))
        return (None if full is None else full.double().numpy()), (None if third is None else third.double().numpy())

    def signed(f: int, mask) -> list:
        top, head = st._views(sign(f, mask))
        full, top_again = whole(f, mask)
        if top is not None and top_again is not None and float((top * top_again).sum()) < 0.9999:
            raise SystemExit(f"frame {f}: the top third signed here is not the node's (cosine {float((top * top_again).sum()):.5f})")
        return [None if v is None else v.double().numpy() for v in (top, head, full)]

    R = {"environment": environment(torch), "device": "cpu" if a.cpu else "cuda",
         "sam": "corrected (sam31_corrections)" if a.corrected else "as ComfyUI ships it",
         "frames": {"clip": Path(a.clip).name, "first_frame": a.first_frame, "count": int(a.frames), "size": [a.width, a.height]},
         "settings": {**SUBJECT_TRACK, "max_people": max_people}, "same_person": SAME_PERSON, "margin": a.margin,
         "tables": {label: Path(pair.partition("=")[2]).name for label, pair in zip(tables, a.table)},
         "truth": a.truth, "controls": [], "whole_gallery": {}, "looks": [], "crops": bool(a.crops), "crop_margin": CROP_MARGIN}

    # control: the detections on each table's shown frames are that table's people
    for label, t in tables.items():
        for s in t["shots"]:
            f = int(s["shown_frame"])
            masks, scores = detect(f)
            mine = sorted((list(shot_table._box(m)), round(float(sc), 3)) for m, sc in zip(masks, scores))
            theirs = sorted((list(p["box"]), p["detector_score"]) for p in s["people"])
            off = max((abs(x - y) for (b1, _), (b2, _) in zip(mine, theirs) for x, y in zip(b1, b2)), default=0) if len(mine) == len(theirs) else None
            R["controls"].append({"table": label, "shot": s["shot"], "frame": f, "table_people": len(theirs), "here": len(mine),
                                  "largest_box_difference_px": off,
                                  "largest_score_difference": None if off is None else max((abs(s1 - s2) for (_, s1), (_, s2) in zip(mine, theirs)), default=0.0)})
            print(f"control {label} shot {s['shot']} frame {f}: table {len(theirs)} people, here {len(mine)}, boxes within {off} px", flush=True)

    # the whole-mask gallery: on a gallery frame, the detection most like the gallery's own entry is the subject
    whole_held = {}
    for label, frames in whole_frames.items():
        entries, found = [], []
        for f in frames:
            g = held[label][tables[label]["gallery"]["frames"].index(f)]
            masks, _ = detect(f)
            views = [signed(f, m) for m in masks]
            able = [(float(g[0] @ v[0]), i) for i, v in enumerate(views) if v[0] is not None and g[0] is not None]
            if not able:
                continue
            like, i = max(able)
            found.append({"frame": f, "top_third_likeness_to_the_gallery_entry": round(like, 4)})
            if like >= a.gallery_same:
                entries.append(views[i] + (list(cropped(f, masks[i])) if a.crops else [None, None]))
            print(f"whole gallery {label} frame {f}: the nearest detection is {like:.3f} like the gallery's entry", flush=True)
        whole_held[label] = entries
        R["whole_gallery"][label] = found

    for f in looks:
        masks, scores = detect(f)
        kept, stands = merged(masks, scores)
        order = shot_table.person_order(masks)
        people = []
        for i in kept:
            views = signed(f, masks[i])
            crop_whole, crop_top = cropped(f, masks[i]) if a.crops else (None, None)
            row = {"detection": i, "person": order.index(i) + 1, "stands_for_people": sorted(order.index(j) + 1 for j in stands[i]),
                   "box": list(shot_table._box(masks[i])), "area": int((masks[i] > 0.5).sum()), "detector_score": round(float(scores[i]), 3),
                   "has_head": views[1] is not None, "likeness": {}}
            for label in tables:
                row["likeness"][label] = {"top third": rule(views, held[label], (0,)), "head": rule(views, held[label], (1,)),
                                          "rule": rule(views, held[label], (0, 1)),
                                          "whole": rule([views[2]], [[v[2]] for v in whole_held[label]], (0,)) if whole_held[label] else None}
                row["likeness"][label]["rule and whole"] = (None if row["likeness"][label]["rule"] is None or row["likeness"][label]["whole"] is None
                                                            else min(row["likeness"][label]["rule"], row["likeness"][label]["whole"]))
                if a.crops:
                    held_crops = [v for v in whole_held[label] if len(v) > 4 and v[3] is not None and v[4] is not None]
                    row["likeness"][label]["crop whole"] = rule([crop_whole], [[v[3]] for v in held_crops], (0,)) if held_crops else None
                    row["likeness"][label]["crop top third"] = rule([crop_top], [[v[4]] for v in held_crops], (0,)) if held_crops else None
            people.append(row)
        entry = {"frame": f, "detections": int(masks.shape[0]), "after_merging": len(kept), "people": people, "settled": {}}
        for name in ("top third", "head", "rule", "whole", "rule and whole"):
            entry["settled"][name] = settle({label: [p["likeness"][label][name] for p in people] for label in tables}, a.margin)
        R["looks"].append(entry)
        print(f"look {f}: {int(masks.shape[0])} detections, {len(kept)} people after merging", flush=True)
        Path(a.json).write_text(json.dumps(R, indent=1))
    render_looks(R)


def render_looks(R: dict):
    labels = list(R["tables"])
    print(f"\n{R['frames']['clip']}, {R['frames']['count']} frames from frame {R['frames']['first_frame']} at {R['frames']['size']}; "
          f"SAM 3.1 {R['sam']}, on {R['device']}; margin {R['margin']}; nested detections joined at {R['same_person']}\n")
    print("| control: table | shot | frame | people in the table | here | boxes within (px) | detector scores within |")
    print("|---|---|---|---|---|---|---|")
    for c in R["controls"]:
        print(f"| {c['table']} | {c['shot']} | {c['frame']} | {c['table_people']} | {c['here']} | {c['largest_box_difference_px']} | {_f(c['largest_score_difference'])} |")
    print("\n| frame | person (stands for) | box | detector | head | " + " | ".join(f"{name}: {' / '.join(labels)}" for name in ("top third", "head", "rule", "whole")) + " |")
    print("|---|---|---|---|---|" + "---|" * 4)
    for look in R["looks"]:
        for p in look["people"]:
            cells = [" / ".join(_f(p["likeness"][label][name]) for label in labels) for name in ("top third", "head", "rule", "whole")]
            print(f"| {look['frame']} | {p['person']} ({', '.join(str(v) for v in p['stands_for_people'])}) | {p['box']} | {p['detector_score']} "
                  f"| {'yes' if p['has_head'] else 'no'} | " + " | ".join(cells) + " |")
    print("\n| frame | by | " + " | ".join(f"{label} takes (score, lead over the other subject)" for label in labels) + " |")
    print("|---|---|" + "---|" * len(labels))
    for look in R["looks"]:
        person = {i: p["person"] for i, p in enumerate(look["people"])}
        for name, got in look["settled"].items():
            cells = []
            for label in labels:
                g = got[label]
                who = "NOBODY" if g["takes"] is None else f"person {person[g['takes']]}"
                extra = "" if "score" not in g else f" ({_f(g['score'])}, {_f(g.get('lead'))})"
                cells.append(f"{who}{extra}" + ("" if g["takes"] is not None else f": {g['why']}"))
            print(f"| {look['frame']} | {name} | " + " | ".join(cells) + " |")
    if R.get("crops"):
        print("\nsigned from a crop at one scale: each person's likeness to each gallery (" + " / ".join(labels) + "), frame scale beside crop\n")
        print("| frame | person | box | top third | crop top third | whole | crop whole |")
        print("|---|---|---|---|---|---|---|")
        for look in R["looks"]:
            for p in look["people"]:
                cells = [" / ".join(_f(p["likeness"][label].get(name)) for label in labels) for name in ("top third", "crop top third", "whole", "crop whole")]
                print(f"| {look['frame']} | {p['person']} | {p['box']} | " + " | ".join(cells) + " |")
    print("\nall subjects at once, by the sum (`together`; added after the first run):\n")
    names = ("top third", "head", "rule", "whole", "top third + whole")
    if R.get("crops"):
        names = ("top third", "whole", "top third + whole", "crop top third", "crop whole", "crop top third + crop whole")
    print("| frame | " + " | ".join(f"{name}: who takes whom, lead over the next way" for name in names) + " |")
    print("|---|" + "---|" * len(names))
    for look in R["looks"]:
        person = {i: p["person"] for i, p in enumerate(look["people"])}
        cells = []
        for name in names:
            def score(p, label, name=name):
                like = p["likeness"][label]
                if " + " in name:
                    one, two = name.split(" + ")
                    return None if like.get(one) is None or like.get(two) is None else like[one] + like[two]
                return like.get(name)
            got = together({label: [score(p, label) for p in look["people"]] for label in labels})
            if got is None:
                cells.append("cannot be compared")
            else:
                who = ", ".join(f"{label} " + ("nobody" if c is None else f"person {person[c]}") for label, c in got["takes"].items())
                cells.append(f"{who}; {_f(got['lead'])}")
        print(f"| {look['frame']} | " + " | ".join(cells) + " |")
    if R.get("truth"):
        print("\nthe known answers, as given on the command line: " + "; ".join(R["truth"]))


def cmd_render(a):
    R = json.loads(Path(a.json).read_text())
    (render_looks if "looks" in R else render_corrections if "rows" in R and "tables" in R and isinstance(R["tables"], list) else render_galleries)(R)


def main():
    p = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("galleries")
    g.add_argument("--table", action="append", required=True, help="LABEL=PATH of a shot table; twice or more")
    g.add_argument("--json", default="")
    g.set_defaults(fn=cmd_galleries)
    k = sub.add_parser("looks")
    k.add_argument("--clip", required=True)
    k.add_argument("--first-frame", type=int, required=True, help="the clip's frame that is the load's frame 0")
    k.add_argument("--frames", type=int, required=True)
    k.add_argument("--rate", type=float, default=24.0)
    k.add_argument("--width", type=int, required=True)
    k.add_argument("--height", type=int, required=True)
    k.add_argument("--table", action="append", required=True)
    k.add_argument("--look", required=True, help="frames of the load, comma separated")
    k.add_argument("--truth", action="append", default=[], help="a known answer in words, written into the data as given")
    k.add_argument("--margin", type=float, default=0.03, help="a subject's lead over every other subject on its candidate; default the node's REGAIN_MARGIN")
    k.add_argument("--gallery-every", type=int, default=2, help="every n-th gallery frame is used for the whole-mask gallery")
    k.add_argument("--gallery-same", type=float, default=0.95, help="a detection is the gallery frame's subject at this top-third likeness to its entry")
    k.add_argument("--max-people", type=int, default=0)
    k.add_argument("--corrected", action="store_true")
    k.add_argument("--crops", action="store_true", help="also sign every person from a square crop at one scale")
    k.add_argument("--cpu", action="store_true")
    k.add_argument("--json", required=True)
    k.set_defaults(fn=cmd_looks)
    c = sub.add_parser("corrections")
    c.add_argument("--table", action="append", required=True, help="LABEL=PATH of a shot table; one per tracker of the preview")
    c.add_argument("--out", required=True, help="the file for the gate")
    c.add_argument("--unsure", choices=("ask", "take"), default="ask", help="what an unsure row becomes; `take` takes the top way anyway")
    c.add_argument("--suggest-lead", type=float, default=None,
                   help="the lead at or over which a differing row is written with its correction filled in; unset, every such row asks with candidates")
    c.add_argument("--first-frame", type=int, default=None, help="the source's frame that is the load's frame 0; written into the file as given")
    c.set_defaults(fn=cmd_corrections)
    r = sub.add_parser("render")
    r.add_argument("--json", required=True)
    r.set_defaults(fn=cmd_render)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
