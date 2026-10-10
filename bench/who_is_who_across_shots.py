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
    render      prints the tables of a `looks`, `galleries` or `corrections` JSON.

    <python> bench/who_is_who_across_shots.py galleries --table L=<shots.json> --table S=<shots.json> [--json J]
    <python> bench/who_is_who_across_shots.py looks --clip C --first-frame F --frames N --width W --height H \\
             --table L=<shots.json> --table S=<shots.json> --look 406,410,... --truth 'S@406-434=small' ... --json J [--cpu]
    <python> bench/who_is_who_across_shots.py corrections --table lead=<shots.json> --table second=<shots.json> --out J [--unsure take]
    <python> bench/who_is_who_across_shots.py render --json J

THREE VIEWS. The node signs a person in two places, the top third of the mask and the head. `looks` adds the WHOLE
mask. A table's gallery has no whole-mask signature, so for that view the gallery is made here: on the gallery's own
frames the detection most like the gallery entry is taken as the subject, and its whole mask is signed.

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


def read_tables(pairs: list[str]) -> dict[str, dict]:
    out = {}
    for pair in pairs:
        label, _, path = pair.partition("=")
        if not label or not path:
            raise SystemExit(f"--table wants LABEL=PATH; got {pair!r}")
        out[label] = json.loads(Path(path).read_text())
        if not (out[label].get("gallery") or {}).get("signatures"):
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


def settle_tables(tables: dict[str, dict], least_lead: float | None = None, unsure: str = "ask", first_source_frame: int | None = None) -> dict:
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
        if not [v for v in t["gallery"]["signatures"] if v and v[0] is not None] or not [v for v in t["gallery"]["whole"] if v is not None]:
            raise SystemExit(f"{label}: this shot table carries no gallery of its subject (nobody was picked, or the picked shot was "
                             "settled by hand), so nobody can be compared with that subject")
    top = {label: [v[0] for v in t["gallery"]["signatures"] if v and v[0] is not None] for label, t in tables.items()}
    whole = {label: [v for v in t["gallery"]["whole"] if v is not None] for label, t in tables.items()}
    least = S.TOGETHER_LEAD if least_lead is None else least_lead
    rows = []
    for n in range(len(first["shots"])):
        by_frame: dict[int, list] = {}
        for label in labels:
            shot = tables[label]["shots"][n]
            for frame, people in [(shot["shown_frame"], shot["people"])] + [(look["frame"], look["people"]) for look in shot.get("looks", [])]:
                signed = [p for p in people if (p.get("signatures") or [None])[0] is not None and p.get("whole_signature") is not None]
                if people and len(signed) == len(people) and len(people) > len(by_frame.get(int(frame), [])):
                    by_frame[int(frame)] = people
        frames = sorted(by_frame)
        names = S.same_people([[_xyxy(p["box"]) for p in by_frame[f]] for f in frames])
        looks = []
        for f, row in zip(frames, names):
            looks.append({label: {name: (None if _best(p["signatures"][0], top[label]) is None or _best(p["whole_signature"], whole[label]) is None
                                         else _best(p["signatures"][0], top[label]) + _best(p["whole_signature"], whole[label]))
                                  for name, p in zip(row, by_frame[f])} for label in labels})
        answer = S.hand_out(looks)
        verdict, why = S.sure(answer, least)
        if verdict == S.ASK and unsure == "take" and answer["takes"] is not None:
            verdict, why = S.TAKE, f"UNSURE AND TAKEN BY THE SWITCH ({why})"
        on_frame = {f: {name: _xyxy(p["box"]) for name, p in zip(row, by_frame[f])} for f, row in zip(frames, names)}
        trackers, given = {}, []
        for label in labels:
            shot = tables[label]["shots"][n]
            called = {"state": shot["subject"]["state"], "person": shot["subject"]["person"]}
            says, correction, tile = "cannot say", "", int(shot["shown_frame"])
            if answer["takes"] is not None:
                name = answer["takes"][label]
                if name is None:
                    says = None
                    correction = "" if called["person"] is None else f"shot {shot['shot']}: none"
                elif tile not in on_frame:
                    says = "cannot say: this tracker's tile frame is not among the looks read"
                elif name not in on_frame[tile]:
                    says = "cannot say: that person is not on this tracker's tile frame"
                else:
                    # the same frame, so the same person's two boxes are one box: this tracker's own number for them
                    overlap, person = max(((S.box_overlap(_xyxy(p["box"]), on_frame[tile][name]), p["person"]) for p in shot["people"]),
                                          default=(0.0, None))
                    if overlap >= 0.9:
                        says = person
                        correction = "" if called["person"] == says else f"shot {shot['shot']}: person {says}"
                        given.append((label, tile, on_frame[tile][name]))
                    else:
                        says = "cannot say: that person is not numbered on this tracker's tile"
            trackers[label] = {"called": called["state"], "called_person": called["person"], "the_sum_says": says,
                               "correction": correction, "typed_in_this_table": str(shot.get("corrected", "") or ""),
                               "tile_frame": tile}
        fault = [(a[0], b[0]) for i, a in enumerate(given) for b in given[i + 1:] if a[1] == b[1] and S.box_overlap(a[2], b[2]) >= 0.9]
        cannot = answer["takes"] is None or any(isinstance(t["the_sum_says"], str) for t in trackers.values())
        if fault:
            for t in trackers.values():
                t["correction"] = ""
            cannot, why = True, f"A FAULT OF THIS TOOL: {fault[0][0]} and {fault[0][1]} were handed one person on one tile frame; nothing is written to type"
        status = UNRESOLVED if cannot else DIFFERS if any(t["correction"] for t in trackers.values()) else AGREES
        # everybody seen in the shot, for a reader that must tell a person from a thing that never moves
        seen = {}
        for f, row in zip(frames, names):
            for name, p in zip(row, by_frame[f]):
                box = _xyxy(p["box"])
                seen.setdefault(name, []).append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, (box[2] - box[0]) * (box[3] - box[1])))
        people_seen = [{"name": name, "looks_seen": len(v),
                        "centre_moved_px": round(max(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a in v for b in v), 1),
                        "box_area_px": [int(min(x[2] for x in v)), int(max(x[2] for x in v))],
                        "left_out_as_seldom_seen": name in answer["left_out"],
                        "handed_to": next((label for label in labels if answer["takes"] is not None and answer["takes"][label] == name), None)}
                       for name, v in sorted(seen.items())]
        rows.append({"shot": first["shots"][n]["shot"], "frames": [first["shots"][n]["first_frame"], first["shots"][n]["last_frame"]],
                     "status": status, "verdict": verdict, "why": why,
                     "looks_with_signatures": frames, "looks_used": answer["looks_used"], "people": answer["people"],
                     "lead": answer["lead"], "trackers": trackers, "people_seen": people_seen})
    return {"what": "every shot of one load decided for all its trackers' subjects at once; a file for the gate, not a job",
            "status_is": f"{AGREES}: the sum would change nothing. {DIFFERS}: it would, and `correction` is the text to type into that "
                         f"tracker. {UNRESOLVED}: it cannot say. A row agrees once its correction is typed and the preview run again",
            "frames_are": "the load's own frame numbers" + ("" if first_source_frame is None else f"; the load's first frame is the source's {int(first_source_frame)}, as given"),
            "first_source_frame": first_source_frame,
            "tables": labels, "least_lead": least, "unsure": unsure, "rows": rows}


def cmd_corrections(a):
    tables = read_tables(a.table)
    out = settle_tables(tables, unsure=a.unsure, first_source_frame=a.first_frame)
    out["files"] = {label: Path(pair.partition("=")[2]).name for label, pair in zip(tables, a.table)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    render_corrections(out)


def render_corrections(out: dict):
    labels = out["tables"]
    print(f"{out['what']}\n{out['status_is']}\nleast lead to take without a person: {out['least_lead']}; unsure rows: {out['unsure']}; {out['frames_are']}\n")
    print("| shot | frames | status | looks used | people | lead | " + " | ".join(f"{label}: called, the sum says" for label in labels) + " | without a person | to type |")
    print("|---|---|---|---|---|---|" + "---|" * len(labels) + "---|---|")
    for row in out["rows"]:
        cells = []
        for label in labels:
            t = row["trackers"][label]
            called = t["called"] + ("" if t["called_person"] is None else f" (person {t['called_person']})")
            says = "nobody" if t["the_sum_says"] is None else (f"person {t['the_sum_says']}" if isinstance(t["the_sum_says"], int) else t["the_sum_says"])
            cells.append(f"{called}; {says}")
        typed = "; ".join(f"{label}: `{row['trackers'][label]['correction']}`" for label in labels if row["trackers"][label]["correction"]) or "nothing"
        print(f"| {row['shot']} | {row['frames'][0]}-{row['frames'][1]} | {row['status']} | {row['looks_used']} | {row['people']} | {_f(row['lead'])} | "
              + " | ".join(cells) + f" | {row['verdict']}: {row['why']} | {typed} |")


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
         "truth": a.truth, "controls": [], "whole_gallery": {}, "looks": []}

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
                entries.append(views[i])
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
            row = {"detection": i, "person": order.index(i) + 1, "stands_for_people": sorted(order.index(j) + 1 for j in stands[i]),
                   "box": list(shot_table._box(masks[i])), "area": int((masks[i] > 0.5).sum()), "detector_score": round(float(scores[i]), 3),
                   "has_head": views[1] is not None, "likeness": {}}
            for label in tables:
                row["likeness"][label] = {"top third": rule(views, held[label], (0,)), "head": rule(views, held[label], (1,)),
                                          "rule": rule(views, held[label], (0, 1)),
                                          "whole": rule([views[2]], [[v[2]] for v in whole_held[label]], (0,)) if whole_held[label] else None}
                row["likeness"][label]["rule and whole"] = (None if row["likeness"][label]["rule"] is None or row["likeness"][label]["whole"] is None
                                                            else min(row["likeness"][label]["rule"], row["likeness"][label]["whole"]))
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
    print("\nall subjects at once, by the sum (`together`; added after the first run):\n")
    names = ("top third", "head", "rule", "whole", "top third + whole")
    print("| frame | " + " | ".join(f"{name}: who takes whom, lead over the next way" for name in names) + " |")
    print("|---|" + "---|" * len(names))
    for look in R["looks"]:
        person = {i: p["person"] for i, p in enumerate(look["people"])}
        cells = []
        for name in names:
            def score(p, label, name=name):
                like = p["likeness"][label]
                if name == "top third + whole":
                    return None if like["top third"] is None or like["whole"] is None else like["top third"] + like["whole"]
                return like[name]
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
    k.add_argument("--cpu", action="store_true")
    k.add_argument("--json", required=True)
    k.set_defaults(fn=cmd_looks)
    c = sub.add_parser("corrections")
    c.add_argument("--table", action="append", required=True, help="LABEL=PATH of a shot table; one per tracker of the preview")
    c.add_argument("--out", required=True, help="the file for the gate")
    c.add_argument("--unsure", choices=("ask", "take"), default="ask", help="what an unsure row becomes; `take` takes the top way anyway")
    c.add_argument("--first-frame", type=int, default=None, help="the source's frame that is the load's frame 0; written into the file as given")
    c.set_defaults(fn=cmd_corrections)
    r = sub.add_parser("render")
    r.add_argument("--json", required=True)
    r.set_defaults(fn=cmd_render)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
