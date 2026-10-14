#!/usr/bin/env python3
"""Loads to propose on a stretch of one subject: where the part clears the capture's floor, frame by frame, and the runs
long enough to be a load, each with the node's own zoom. Data only: it proposes; the card and the lead decide.

    <python> bench/propose_runs.py --preview data/<day>_<preview>_job --subject LABEL [--first F --last L] \\
        [--least PX [--least PX ...]] [--pieces largest] [--bridge N] [--share OUTPUT_DIR]

**What it is for.** The lead (2026-10-14): on the far and small stretches of 0-2049 the card chose its loads by eye and by
hand for every stretch (604-851 that hour): which frames the subject's part is big enough to make, where to cut, whether
the node would zoom. This reads the same from the preview's own files, with no card.

**A frame.** The px of the subject's part on the preview's saved part mask (`<preview>_<subject>_parts_*.mkv`, over 127,
at the canvas), the whole mask or, with `--pieces largest`, its largest piece, as the capture reads it with `pieces=largest`;
against each floor (`--least`, the first one the runs are cut by; `capture_masked_run.PART_LEAST_PX` when none is given).

**A run.** The frames that clear the first floor, joined across gaps of at most `--bridge` frames (empty frames inside a
run, listed as MID-RUN), kept when as long as the shortest load (`masked_job.shortest_load`); each with the frames at its
edges that are under the floor next to it (EDGE), and the node's own zoom on its frames: `video_mask.zoom_box` itself,
called on the boxes of the part, not a copy of its arithmetic.

**What it cannot see.** The node boxes what it is handed at plan time, the load's region (the part grown by its margin),
so near `ZOOM_SMALLEST` its zoom can be a refusal where this says a zoom (said with each run). The capture's grade
(`part_grades.json`) can empty a frame this keeps (a held part, a doubted piece): the capture is made on the card. Whether
the part is the right person: a PICKED tracker's part may be somebody else's. The picture.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "workflows"))

#: Empty frames a run is joined across, a frame at a time. A STARTING VALUE from one case: the lead approved 751-778 of
#: shot_c_0604b's lead with 761 empty inside it (mrcircle, 2026-10-14).
BRIDGE = 1


def node_zoom():
    """The node's own `video_mask.zoom_box`, loaded as the pack's module without the server (one copy of the arithmetic)."""
    comfy_root = REPO.parent.parent
    if str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))
    if "h3pack" not in sys.modules:
        pkg = types.ModuleType("h3pack")
        pkg.__path__ = [str(REPO)]
        sys.modules["h3pack"] = pkg
    return importlib.import_module("h3pack.video_mask")


def part_frames(mask: np.ndarray, largest: bool) -> tuple[np.ndarray, np.ndarray]:
    """(px a frame, box a frame as x0, y0, x1, y1 with -1 where there is none) of a part mask [n, h, w], whole or its largest piece."""
    from scipy import ndimage
    px, boxes = np.zeros(len(mask), int), np.full((len(mask), 4), -1, int)
    for f, m in enumerate(mask):
        if largest and m.any():
            lab, n = ndimage.label(m)
            m = lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))
        px[f] = int(m.sum())
        ys, xs = np.nonzero(m)
        if len(xs):
            boxes[f] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return px, boxes


def runs_of(ok: np.ndarray, bridge: int, least_frames: int) -> list[tuple[int, int]]:
    """The runs of True, joined across gaps of at most `bridge` False, as (first, last) indices, kept when `least_frames` long."""
    spans, start = [], None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        if not v and start is not None:
            spans.append([start, i - 1])
            start = None
    if start is not None:
        spans.append([start, len(ok) - 1])
    joined = []
    for s in spans:
        if joined and s[0] - joined[-1][1] - 1 <= bridge:
            joined[-1][1] = s[1]
        else:
            joined.append(s)
    return [(a, b) for a, b in joined if b - a + 1 >= least_frames]


def propose(px: np.ndarray, boxes: np.ndarray, first: int, floors: list[int], bridge: int, least_frames: int, size: tuple[int, int], zoom_box) -> dict:
    """Per frame the px against each floor, and the runs the first floor gives, each with its edges, its mid-run empty frames and the
    node's zoom."""
    import torch
    floor = floors[0]
    ok = px >= floor
    runs = []
    for a, b in runs_of(ok, bridge, least_frames):
        inside = [first + i for i in range(a, b + 1) if not ok[i]]
        edge_before = [first + i for i in range(max(a - 3, 0), a) if not ok[i]]
        edge_after = [first + i for i in range(b + 1, min(b + 4, len(ok))) if not ok[i]]
        box, why = zoom_box(torch.tensor(boxes[a:b + 1]), size[0], size[1], size[0], size[1])
        zoom = {"zooms": round(size[0] / (box[2] - box[0]), 2), "box": list(box), "set_by": why} if box is not None else {"zooms": None, "why": why}
        runs.append({"frames": [first + a, first + b], "count": b - a + 1, "mid_run_under_the_floor": inside, "no_part_mid_run": [f for f in inside if px[f - first] == 0],
                     "edge_frames_under_the_floor_next_to_it": edge_before + edge_after, "zoom": zoom,
                     "px_median": int(np.median(px[a:b + 1])), "px_least": int(px[a:b + 1].min())})
    covered = {f for r in runs for f in range(r["frames"][0], r["frames"][1] + 1)}
    return {"floors": floors, "bridge": bridge, "least_frames": least_frames,
            "frames": [{"frame": first + i, "px": int(v), **{f"clears_{fl}": bool(v >= fl) for fl in floors}} for i, v in enumerate(px)],
            "runs": runs, "left_as_the_original": spans_of([first + i for i in range(len(px)) if first + i not in covered])}


def spans_of(frames: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--preview", required=True, help="the preview's job folder (data/<day>_<preview>_job)")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--first", type=int, help="the stretch's first source frame (the preview's when left out)")
    ap.add_argument("--last", type=int, help="the stretch's last source frame (the preview's when left out)")
    ap.add_argument("--least", type=int, action="append", default=[], help="a floor in px of the canvas; the first cuts the runs; repeatable")
    ap.add_argument("--pieces", choices=("all", "largest"), default="all", help="the part's largest piece a frame (as pieces=largest), or the whole mask")
    ap.add_argument("--bridge", type=int, default=BRIDGE, help=f"empty frames a run is joined across ({BRIDGE}, a starting value)")
    ap.add_argument("--share", default="/mnt/hub/ai/img/output", help="the server's output folder the preview wrote to")
    ap.add_argument("--json", help="also write the proposal as JSON here (a new file)")
    a = ap.parse_args(argv)
    import capture_masked_run as cap
    import masked_job
    from _lib.frames import read_mask
    folder = Path(a.preview)
    job = json.loads((folder / "job.json").read_text())["job"]
    files = json.loads((folder / "preview.queued.json").read_text()).get("files") or []
    found = [f for f in files if f"_{a.subject}_parts_" in Path(f).name and f.endswith(".mkv")]
    if len(found) != 1:
        raise SystemExit(f"the preview {job['name']} saved {len(found)} part video(s) of {a.subject!r} where one is meant")
    (w, h), pf, n = job["canvas"], int(job["first"]), int(job["frames"])
    first, last = a.first if a.first is not None else pf, a.last if a.last is not None else pf + n - 1
    if not (pf <= first <= last <= pf + n - 1):
        raise SystemExit(f"{first}-{last} is not inside the preview's {pf}-{pf + n - 1}")
    floors = a.least or [int(cap.PART_LEAST_PX)]
    mask = read_mask(str(Path(a.share) / found[0]), (w, h), pf, first, last - first + 1)[0]
    px, boxes = part_frames(mask, a.pieces == "largest")
    least_frames = int(masked_job.shortest_load()[1])
    out = propose(px, boxes, first, floors, a.bridge, least_frames, (w, h), node_zoom().zoom_box)
    out.update(preview=job["name"], subject=a.subject, stretch=[first, last], pieces=a.pieces, part_file=found[0],
               cannot_see="the node boxes the load's region (part plus margin), so near ZOOM_SMALLEST a zoom here can be a refusal there; the capture's "
                          "grade can empty a frame this keeps; whether the part is the right person; the picture")
    print(f"{job['name']} {a.subject} {first}-{last}, the part {'(largest piece) ' if a.pieces == 'largest' else ''}against {', '.join(map(str, floors))} px; "
          f"runs of {least_frames} frames or more, joined across {a.bridge} empty frame(s):")
    for r in out["runs"]:
        z = r["zoom"]
        print(f"  PROPOSE {r['frames'][0]}-{r['frames'][1]} ({r['count']} frames), px median {r['px_median']}, least {r['px_least']}; "
              + (f"zooms {z['zooms']}x ({z['set_by']})" if z["zooms"] else f"NOT ZOOMED: {z['why']}")
              + (f"; under the floor mid-run: {r['mid_run_under_the_floor']}" if r["mid_run_under_the_floor"] else "")
              + (f"; at its edges: {r['edge_frames_under_the_floor_next_to_it']}" if r["edge_frames_under_the_floor_next_to_it"] else ""))
    print("  left as the original: " + (", ".join(f"{p}-{q}" for p, q in out["left_as_the_original"]) or "none"))
    for fl in floors[1:]:
        print(f"  at {fl} px: {sum(1 for f in out['frames'] if f[f'clears_{fl}'])} of {len(out['frames'])} frames clear it")
    print(f"  CANNOT SEE: {out['cannot_see']}")
    if a.json:
        if Path(a.json).exists():
            raise SystemExit(f"refused: {a.json} is there; a kept output is not replaced")
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
