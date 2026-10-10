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
    render      prints the tables of a `looks` or `galleries` JSON.

    <python> bench/who_is_who_across_shots.py galleries --table L=<shots.json> --table S=<shots.json> [--json J]
    <python> bench/who_is_who_across_shots.py looks --clip C --first-frame F --frames N --width W --height H \\
             --table L=<shots.json> --table S=<shots.json> --look 406,410,... --truth 'S@406-434=small' ... --json J [--cpu]
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
    if R.get("truth"):
        print("\nthe known answers, as given on the command line: " + "; ".join(R["truth"]))


def cmd_render(a):
    R = json.loads(Path(a.json).read_text())
    (render_looks if "looks" in R else render_galleries)(R)


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
    r = sub.add_parser("render")
    r.add_argument("--json", required=True)
    r.set_defaults(fn=cmd_render)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
