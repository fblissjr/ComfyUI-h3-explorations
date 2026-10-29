#!/usr/bin/env python3
"""3-way replay of all `--crop cranium` head references (`pre-filter` vs `landed b7875bf1` vs `current cranium_box`).

Compares every `*_head_reference_*.json` record under `$H3_OUTPUT_DIR/Video/<video>/references/` whose `crop.cranium`
is a dict across three rules:
1. `pre-filter`: the original unbounded `grey_head` rule with no score floor and no face-scale filter.
2. `landed b7875bf1`: `CRANIUM_SCORE_FLOOR = 0.30` and `CRANIUM_FACE_OUTLIER = 2.50` with unbounded `grey_head`.
3. `current`: 4-corner SAM 3.1 `head_track` containment (`CRANIUM_HEAD_TRACK_GROW = 0.50`) + `grey_head` bounded to the
   grown head-track window (`win[k]`), with one active sizing median (`face`) used everywhere and a `2.0x` face-scale
   fallback (`CRANIUM_FACE_OUTLIER = 2.00`) when no `head_track` is present.

Writes the full per-record comparison table as JSON to `data/<video>/cranium_replay_<video>_<stamp>.json` (or `--out`
when explicitly given) with `writer_stamp` and the rule constants in force, and prints summary statistics and the path
written.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime as dt
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from _lib import face_points as fp  # noqa: E402
from _lib import writer_stamp  # noqa: E402
from _lib.frames import read_mask, stream  # noqa: E402
import crop_reference_to_head as cr  # noqa: E402


def _legacy_cranium_box(
    points: dict,
    size: tuple[int, int],
    room: float,
    frames: np.ndarray,
    first: int,
    leave_out_frames: list[int],
    raw_pairs: list[tuple[np.ndarray, int]],
    *,
    score_floor: float,
    face_outlier: float,
) -> dict:
    """Evaluate the pre-filter (`score_floor=-1, face_outlier=inf`) or landed `b7875bf1` (`0.30, 2.50`) rule."""
    w, h = size
    idx = points["index"]
    face = float(np.median([fp.face_height(p, idx) for p in points["keypoints"]]))
    scale_r = room / cr.HEAD_ROOM if cr.HEAD_ROOM else 1.0
    c_room, neck = cr.CRANIUM_ROOM * scale_r, cr.CRANIUM_NECK * scale_r
    each, heads = [], []
    for p, (head, _out) in zip(points["keypoints"], raw_pairs):
        fb = cr.face_box(p, idx)
        ys, xs = np.nonzero(head)
        top = [fb[0], fb[1], fb[2]] if not len(xs) else [min(fb[0], xs.min()), min(fb[1], ys.min()), max(fb[2], xs.max() + 1)]
        chin = max(float(fb[3]), float(p[idx["tip_of_chin"], 1]))
        each.append([top[0] - c_room * face, top[1] - c_room * face, top[2] + c_room * face, chin + neck * face])
        heads.append(head)
    each = np.array(each, float)
    excluded = sorted({int(f) for f in (leave_out_frames or []) if first <= int(f) < first + len(each)})
    excl_set = set(excluded)
    kept = np.array([(first + k) not in excl_set for k in range(len(each))], dtype=bool)
    fids = fp.face_ids(idx)
    scores = points.get("scores")
    fhs = np.array([fp.face_height(p, idx) for p in points["keypoints"]], dtype=float)
    scored = (
        np.array([float(np.median(scores[k, fids])) >= score_floor for k in range(len(each))], dtype=bool)
        if score_floor > 0 and scores is not None and len(scores) == len(each) and len(fids)
        else np.ones(len(each), dtype=bool)
    )
    seen = np.array([bool(np.any(hd)) for hd in heads], dtype=bool)
    active_fhs = fhs[kept & seen & scored & (fhs > 0)]
    if not len(active_fhs):
        active_fhs = fhs[kept & seen & (fhs > 0)]
    face_ref = float(np.median(active_fhs)) if len(active_fhs) else face
    in_scale = (fhs <= face_outlier * face_ref) if (face_ref > 0 and np.isfinite(face_outlier)) else np.ones(len(each), dtype=bool)
    good = seen & scored & in_scale
    use = (good & kept) if (good & kept).any() else (seen & kept)
    sized = each[use] if use.any() else (each[good] if good.any() else (each[seen] if seen.any() else each))
    want = np.concatenate([sized[:, :2].min(0), sized[:, 2:].max(0)])
    need_w, need_h = float(want[2] - want[0]), float(want[3] - want[1])
    least_w = w / cr.CRANIUM_MOST
    box, _ = cr.in_the_canvas(max(need_w, least_w), need_h, float((want[0] + want[2]) / 2), float((want[1] + want[3]) / 2), size)
    x0, y0, x1, y1 = box
    edge = []
    for k, head in enumerate(heads):
        ys, xs = np.nonzero(head)
        if not len(xs):
            continue
        if use.any() and (first + k) in excl_set:
            continue
        if xs.min() <= x0 or ys.min() <= y0 or xs.max() >= x1 - 1 or ys.max() >= y1 - 1:
            edge.append(first + k)
    return {
        "box": box,
        "enlarges": round(w / (x1 - x0), 3),
        "face_height_px_median": round(face, 1),
        "edge_frames": edge,
    }


def eval_one(p: str) -> dict | None:
    try:
        d = json.loads(Path(p).read_text())
    except Exception:
        return None
    if not isinstance((d.get("crop") or {}).get("cranium"), dict):
        return None
    inp = d.get("inputs") or {}
    bmv = inp.get("body_model_video")
    pts_path = inp.get("points_filled") or inp.get("points")
    if not bmv or not pts_path or not Path(bmv).is_file() or not Path(pts_path).is_file():
        return None
    first, last = d["frames"]
    count = last - first + 1
    w, h = d.get("size") or [1024, 768]
    mf = int(inp.get("its_first_frame_is_source_frame", first))
    pf = int(inp.get("their_first_row_is_source_frame", first))
    P = fp.load(pts_path, None)
    off = first - pf
    pts = dict(P, keypoints=P["keypoints"][off:off + count], scores=P["scores"][off:off + count])
    frames = np.stack(list(stream(bmv, (w, h), first - mf, count, pix="rgb24")))
    lo = list(d["crop"]["cranium"].get("frames_left_out_of_the_box") or [])

    ht, htf, _how = cr.discover_head_track(inp.get("points") or pts_path, bmv, mf)
    masks = None
    if ht and htf is not None and Path(ht).is_file():
        try:
            _, _, have_ht = cr.probe(ht)
            if 0 <= first - htf and first - htf + count <= have_ht:
                masks = read_mask(ht, (w, h), htf, first, count)[0]
        except Exception:
            masks = None

    idx = pts["index"]
    raw_pairs = [cr.grey_head(frame, pt, idx) for pt, frame in zip(pts["keypoints"], frames)]
    pre = _legacy_cranium_box(pts, (w, h), cr.HEAD_ROOM, frames, first, lo, raw_pairs, score_floor=-1.0, face_outlier=float("inf"))
    lnd = _legacy_cranium_box(pts, (w, h), cr.HEAD_ROOM, frames, first, lo, raw_pairs, score_floor=0.30, face_outlier=2.50)
    cur = cr.cranium_box(pts, (w, h), cr.HEAD_ROOM, frames, first, leave_out_frames=lo, head_track=masks)
    c = cur["cranium"]
    return {
        "name": Path(p).parent.name,
        "frames": [first, last],
        "has_ht": bool(c.get("head_track_used")),
        "disk_box": d.get("box"),
        "disk_enl": d["crop"]["cranium"].get("the_one_box_enlarges"),
        "pre_box": pre["box"],
        "pre_enl": pre["enlarges"],
        "pre_edge": len(pre["edge_frames"]),
        "lnd_box": lnd["box"],
        "lnd_enl": lnd["enlarges"],
        "lnd_edge": len(lnd["edge_frames"]),
        "new_box": cur["box"],
        "new_enl": c["the_one_box_enlarges"],
        "new_edge": len(c["frames_the_grey_head_reaches_the_box_edge"]),
        "new_edge_frames": c["frames_the_grey_head_reaches_the_box_edge"],
        "new_unsized_edge": len(c.get("frames_not_sizing_whose_grey_head_reaches_the_box_edge") or []),
        "new_unsized_edge_frames": c.get("frames_not_sizing_whose_grey_head_reaches_the_box_edge") or [],
        "new_raw_edge": len(c.get("frames_the_raw_grey_head_reaches_the_box_edge") or []),
        "new_raw_edge_frames": c.get("frames_the_raw_grey_head_reaches_the_box_edge") or [],
        "new_picture_edge_frames": c.get("frames_the_grey_head_reaches_the_picture_edge") or [],
        "new_face_median": cur["face_height_px_median"],
        "off_trk": len(c.get("frames_off_the_head_track") or []),
        "off_trk_frames": c.get("frames_off_the_head_track") or [],
        "no_trk": len(c.get("frames_with_no_head_track") or []),
        "no_trk_frames": c.get("frames_with_no_head_track") or [],
        "over_scale": len(c.get("frames_over_the_face_scale") or []),
        "over_scale_frames": c.get("frames_over_the_face_scale") or [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--video", default="fun", help="video slug under $H3_OUTPUT_DIR/Video/<video>/references/ (default: fun)")
    ap.add_argument("--workers", type=int, default=12, help="parallel worker processes (default: 12)")
    ap.add_argument(
        "--out",
        default=None,
        help="explicit path to write full JSON table (default: data/<video>/cranium_replay_<video>_<stamp>.json)",
    )
    args = ap.parse_args()
    os.chdir(REPO)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    out_p = Path(args.out) if args.out else (REPO / "data" / args.video / f"cranium_replay_{args.video}_{stamp}.json")
    out_root = Path(os.environ.get("H3_OUTPUT_DIR", "/mnt/hub/ai/img/output"))
    pattern = str(out_root / "Video" / args.video / "references" / "*" / "*head_reference*.json")
    ps = [p for p in sorted(glob.glob(pattern)) if "_plain_cut" not in p and "_source_cut" not in p and "_points" not in p]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = [r for r in ex.map(eval_one, ps) if r is not None]
    out_p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "h3_cranium_replay/1",
        "video": args.video,
        "at": stamp,
        "writer": writer_stamp(__file__),
        "record_count": len(res),
        "scale_fallback_count": sum(1 for r in res if not r["has_ht"]),
        "rules_compared": {
            "pre_filter": {"score_floor": None, "face_outlier": None, "head_track_bound": False},
            "landed_b7875bf1": {"score_floor": 0.30, "face_outlier": 2.50, "head_track_bound": False},
            "current": {
                "score_floor": None,
                "head_track_grow": cr.CRANIUM_HEAD_TRACK_GROW,
                "face_outlier_fallback": cr.CRANIUM_FACE_OUTLIER,
                "cranium_room": cr.CRANIUM_ROOM,
                "cranium_neck": cr.CRANIUM_NECK,
                "cranium_most": cr.CRANIUM_MOST,
            },
        },
        "records": res,
    }
    out_p.write_text(json.dumps(payload, indent=2) + "\n")
    changed_pre = [r for r in res if abs(r["new_enl"] - r["pre_enl"]) >= 0.01 or r["new_box"] != r["pre_box"]]
    changed_lnd = [r for r in res if abs(r["new_enl"] - r["lnd_enl"]) >= 0.01 or r["new_box"] != r["lnd_box"]]
    smaller_pre = [r for r in res if r["new_enl"] - r["pre_enl"] <= -0.05]
    print(
        f"Replayed {len(res)} cranium records -> {out_p}\n"
        f"Changed vs pre-filter: {len(changed_pre)}/{len(res)}; "
        f"Changed vs landed b7875bf1: {len(changed_lnd)}/{len(res)}; "
        f"Smaller than pre-filter by >= 0.05x: {len(smaller_pre)}/{len(res)}"
    )
    for r in sorted(res, key=lambda x: -abs(x["new_enl"] - x["pre_enl"]))[:35]:
        print(
            f"  {r['name']:26s} | pre {r['pre_enl']:5.3f}x (e={r['pre_edge']:2d}) | "
            f"lnd {r['lnd_enl']:5.3f}x (e={r['lnd_edge']:2d}) | "
            f"new {r['new_enl']:5.3f}x (e={r['new_edge']:2d}) | "
            f"off={r['off_trk']:3d} no_trk={r['no_trk']:3d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
