#!/usr/bin/env python3
"""At each look after a loss: is the subject among the people the detector returns, and would a rule take them back?

Today's Subject Track (`subject_track.py`) looks again every few frames after it loses its subject, asks SAM 3 for at
most `max_people` people on the whole frame, compares each with the subject's gallery, and takes the best one when it is
over a line and leads the next by a margin. On a hard crowd stretch (`bench/results/2026-10-07_subject_track_under_nudge.md`)
most looks were refused, every look returned exactly the sixteen people asked for, and the frame holds more than that.
So two things can be wrong at a refused look: the subject was not among the people returned, or they were and the
likeness could not tell them from the next person. This tool takes the node's own run, and at every look it made after
a loss asks again six ways. `max_people` is one input of the node and sets both the people and the heads asked for (a
person with no head found scores as no match, so the cap acts twice), so "32" and "64" raise both:

    16, the frame                today's node; must reproduce the node's own best and next, or the row says so
    32, the frame                `max_people` 32
    64, the frame                `max_people` 64
    64 people, 16 heads          a diagnostic: which of the two caps a gain comes from
    16, a crop                   16 people and 16 heads asked on a crop around where the subject's box last was
    16, the centre               16 and 16 asked on a FIXED region, the middle half of the frame in each direction (the
                                 owner's suggestion for a clip whose subject is always near the middle). Detected on the
                                 crop, not filtered from a whole-frame detect: filtering would not spend the count there.

and, for each, decides two ways: today's rule (`subject_track.clear_best`), and `subject_tracks.take_back`, which
breaks a call too close by likeness with where the subject last was. It runs on the frames as fed and again with every
value moved one level of 255.

    <python> bench/subject_regain_looks.py run --clip C --second S --seconds T --width W --rate R --pick P --json J [--corrected]
    <python> bench/subject_regain_looks.py render --json J
    <python> bench/subject_regain_looks.py replay --json J [--track T]      no model: the recorded looks through `subject_tracks.take_back_by_place`

THE READING, WRITTEN BEFORE THE FIRST RUN (2026-10-07). A higher count, the crop, or the centre region is worth a default or a setting only if
it turns looks that today take nobody into looks that take somebody, in BOTH the plain and the nudged run, and the
person taken is where the subject is next found by today's node (the box taken overlaps the track's next mask's box).
More people over the line with no clearer leader is not a gain. The position rule is worth building only if, at the
looks refused on the lead alone, it takes a candidate in both runs and the two runs' candidates are in the same place.
No arm here can say the person taken IS the subject: there are no labels. One stretch, one clip.

Run from ComfyUI's environment, outside the server, the card free. A clip is named by its file and never by what it shows.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam3_parity_ladder import COMFY, REPO, boot, environment, load_core, nudged   # noqa: E402  (this folder's own tool)

sys.path.insert(0, str(REPO))

#: reasoned: the crop is this many times the subject's last box, around its centre; wide enough to hold a person who moved
#: about their own size, narrow enough that sixteen detections are spent near them.
CROP_TIMES = 3.0
#: reasoned: the fixed region is this share of the frame in each direction, centred; a share, so a zoom does not move it.
CENTRE_SHARE = 0.5
#: read off one run (2026-10-07, the hard crowd stretch, pick `largest`), not set by it: over-the-line candidates at the
#: picked figure's place overlapped its box by 0.39 to 0.59 and everyone else by 0.107 or less. A line between, for the
#: `render` table only; nothing shipped uses it.
AT_THE_PLACE = 0.3
MORE = 64      # inherited: the most core's SAM 3 nodes keep (`comfy/ldm/sam3/tracker.py`'s ceiling; the detect node's count is the phrase's)


def cmd_run(a):
    torch = boot(float32=False)
    import server
    server.PromptServer.instance = types.SimpleNamespace(prompt_queue=None, routes=None)
    sys.path.insert(0, str(COMFY / "custom_nodes" / "ComfyUI-VideoHelperSuite"))
    from videohelpersuite.load_video_nodes import LoadVideoFFmpegPath
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    import comfy.model_management as mm
    import comfy.utils
    from comfy_extras.nodes_sam3 import SAM3_Detect

    import _h3pack.subject_track as st
    import sam31_corrections
    import subject_tracks as S
    from h3_config import SEGMENTER, SUBJECT_TRACK

    loaded = LoadVideoFFmpegPath().load_video(video=a.clip, force_rate=float(a.rate), custom_width=a.width, custom_height=0,
                                              frame_load_cap=int(round(a.seconds * a.rate)), start_time=float(a.second), format="AnimateDiff")[0][..., :3]
    u8 = (loaded.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    del loaded
    n, H, W = (int(v) for v in u8.shape[:3])
    model, clip = load_core(SEGMENTER)
    if a.corrected:
        model, clip = sam31_corrections.corrected(model, clip)
    phrase, head, threshold, few = SUBJECT_TRACK["subject_phrase"], SUBJECT_TRACK["head_phrase"], SUBJECT_TRACK["detection_threshold"], int(SUBJECT_TRACK["max_people"])
    R = {"environment": environment(torch), "frames": {"clip": Path(a.clip).name, "second": a.second, "count": n, "rate": a.rate, "size": [W, H]},
         "sam": "corrected (sam31_corrections)" if a.corrected else "as ComfyUI ships it", "settings": {**SUBJECT_TRACK, "pick": a.pick},
         "device": str(torch.empty(0, device="cuda").device),
         "lines": {"REGAIN_SAME": st.REGAIN_SAME, "REGAIN_MARGIN": st.REGAIN_MARGIN}, "crop_times": CROP_TIMES, "centre_share": CENTRE_SHARE, "runs": {}}

    for name, levels in (("as fed", 0), ("nudged", 1)):
        video = torch.from_numpy(nudged(u8, levels).astype(np.float32) / 255.0)
        detect16, sign16, track = st._sam_callables(model, clip, video, phrase, threshold, few, head)
        detect64, sign64, _ = st._sam_callables(model, clip, video, phrase, threshold, MORE, head)
        detect32, sign32, _ = st._sam_callables(model, clip, video, phrase, threshold, 32, head)
        cond = {k: clip.encode_from_tokens_scheduled(clip.tokenize(st.counted(p, few))) for k, p in (("person", phrase), ("head", head))}
        features = {}
        calls = []

        def tracked(start, end, seed_frame, mask, _track=track):
            """The node's `track`, with each call's own frames put through `subject_tracks.unbroken` as they come back.

            Judged here and not on the finished piece: a later take writes over part of an earlier call's frames
            (`subject_track.py`, the regain), and where two calls meet reads as a jump."""
            out = _track(start, end, seed_frame, mask)
            run = S.unbroken(out > 0.5, int(seed_frame) - int(start))
            lo, hi = S.gallery_span(run)

            def at(i):
                if not 0 <= i < len(run.overlap) or run.overlap[i] is None:
                    return None
                return {"frame": int(i + start), "frames_apart": run.frames_apart[i],
                        **{k: (None if v[i] is None else round(v[i], 3)) for k, v in (("overlap", run.overlap), ("area_ratio", run.area_ratio),
                                                                                       ("box_ratio", run.box_ratio), ("centre_step", run.centre_step))}}
            inside = [v for v in run.overlap[run.first:run.end] if v is not None]
            calls.append({"frames": [int(start), int(end)], "seeded_on": int(seed_frame), "moved_off": S.MOVED_OFF,
                          "unbroken": [int(run.first + start), int(run.end + start)], "stops_before": run.before, "stops_after": run.after,
                          "a_gallery_may_use": [int(lo + start), int(hi + start)], "least_overlap_inside": (round(min(inside), 3) if inside else None),
                          "the_box_on_its_last_gallery_frame": (None if hi <= lo or S.box_of(out[hi - 1]) is None else [round(v, 3) for v in S.box_of(out[hi - 1])]),
                          "the_frame_that_stops_it_before": at(run.first - 1), "its_first_frame": at(run.first),
                          "its_last_frame": at(run.end - 1), "the_frame_that_stops_it_after": at(run.end)})
            return out

        def trunk(f):
            if f not in features:
                mm.load_model_gpu(model)
                x = comfy.utils.common_upscale(video[f:f + 1, ..., :3].movedim(-1, 1), st.TRUNK_SIDE, st.TRUNK_SIDE, "bilinear", crop="disabled")
                out = model.model.diffusion_model.detector.backbone["vision_backbone"].trunk(x.to(device=mm.get_torch_device(), dtype=model.model.get_dtype()))
                features[f] = (out[-1] if isinstance(out, (list, tuple)) else out)[0].to(torch.float32).cpu()
            return features[f]

        def around(box):
            """The crop around `box` (shares of the frame), in pixels."""
            cx, cy, w, h = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H, (box[2] - box[0]) * W, (box[3] - box[1]) * H
            return (int(max(cx - w * CROP_TIMES / 2, 0)), int(max(cy - h * CROP_TIMES / 2, 0)),
                    int(min(cx + w * CROP_TIMES / 2, W)), int(min(cy + h * CROP_TIMES / 2, H)))

        centre = (int(W * (1 - CENTRE_SHARE) / 2), int(H * (1 - CENTRE_SHARE) / 2), int(W * (1 + CENTRE_SHARE) / 2), int(H * (1 + CENTRE_SHARE) / 2))

        def in_a_crop(f, rect):
            """People and heads asked on the pixels of `rect`, pasted back into the frame."""
            x0, y0, x1, y1 = rect
            out = {}
            for k in ("person", "head"):
                got = SAM3_Detect.execute(model, video[f:f + 1, y0:y1, x0:x1], conditioning=cond[k], threshold=float(threshold), individual_masks=True)
                masks = getattr(got, "args", got)[0].to(torch.float32).cpu()
                full = torch.zeros((masks.shape[0], H, W), dtype=torch.float32)
                full[:, y0:y1, x0:x1] = masks
                out[k] = full
            heads = out["head"]

            def sign(frame, mask):
                top = st.head_of(mask, heads)
                return st.signature(trunk(frame), st.top_third(mask)), (None if top is None else st.signature(trunk(frame), top))
            return out["person"], sign, [x0, y0, x1, y1]

        with torch.no_grad():
            steps = st.cut_scores(video, {})
            cuts = st.find_cuts(steps, st.auto_cuts(steps))
            found = st.follow(n, cuts, a.pick, None, None, detect16, sign16, tracked)
            rows = []
            for number, shot in enumerate(found.shots, 1):
                piece = found.pieces.get(shot.start)
                if piece is None or not shot.probes:
                    continue
                # what the node compared with: the subject as the picked shot's track showed them, or this shot's own
                known = list(found.gallery) if shot.picked and found.gallery else [st._views(sign16(g, (piece[g - shot.start] > 0.5).to(torch.float32))) for g in shot.gallery]
                on = (piece > 0.5).flatten(1).any(1)
                # the empty runs the node searched, each by its first frame: (first frame, last frame looked at)
                searched = [(r[0], r[1]) for r in shot.regained] + [(r[0], r[1]) for r in shot.searched]
                # WHERE THE PICKED SUBJECT LAST WAS BEFORE ANYTHING WAS TAKEN AGAIN: the mask before the first empty run. The
                # node's own last place inherits its takes, so after one wrong take a rule by place would defend the wrong
                # person; a tracker with the rule in its loop would have refused that take. This is its last place, for as
                # long as nothing is taken, and the fair anchor on a stretch where the subject stays where they were.
                first_loss = min((first for first, _ in searched), default=None)
                anchor = S.box_of(piece[first_loss - 1 - shot.start]) if first_loss is not None and first_loss > shot.start else None
                pick_box = S.box_of(piece[found.pick_frame - shot.start]) if shot.picked and found.pick_frame is not None else None
                for f, _count, best_then, next_then in ([] if a.track_only else shot.probes):
                    # WHERE THE SUBJECT LAST WAS, AS THE NODE KNEW IT AT THIS LOOK: the mask on the frame before the empty run
                    # being searched began. Not "the last mask before this frame" in the finished track: a take fills its
                    # run backward as well as forward (`subject_track.py`'s regain), so that would read a later take's
                    # fill, the future, as the past. The first version of this tool did, and its crop and position arms
                    # were withdrawn. A take never overwrites a frame outside its own empty run, so this one is safe.
                    began = max((first for first, last in searched if first <= f <= last), default=None)
                    last_box = S.box_of(piece[began - 1 - shot.start]) if began is not None and began > shot.start else None
                    after = on[f - shot.start:].nonzero()
                    next_box = S.box_of(piece[f - shot.start + int(after[0])]) if after.numel() else None
                    many = detect64(f)[0]
                    ways: dict = {"16, the frame": (detect16(f)[0], sign16, None), "32, the frame": (detect32(f)[0], sign32, None),
                                  "64, the frame": (many, sign64, None), "64 people, 16 heads": (many, sign16, None)}
                    if last_box is not None:
                        ways["16, a crop"] = in_a_crop(f, around(last_box))
                    ways["16, the centre"] = in_a_crop(f, centre)
                    row = {"shot": number, "frame": int(f), "the_node_then": {"best": round(float(best_then), 4), "next": round(float(next_then), 4)},
                           "last_box": None if last_box is None else [round(v, 3) for v in last_box], "the_run_searched_began_at": began,
                           "the_place_before_the_first_loss": None if anchor is None else [round(v, 3) for v in anchor],
                           "the_tracks_next_mask_is_at": (int(f + int(after[0])) if after.numel() else None), "ways": {}}
                    for way, (masks, sign, crop) in ways.items():
                        sims = st.gallery_scores(masks, f, known, sign) if masks.shape[0] else []
                        boxes = [S.box_of(m) for m in masks]
                        taken, best, second = st.clear_best(sims)
                        index, reason, detail = S.take_back(sims, boxes, last_box, line=st.REGAIN_SAME, lead=st.REGAIN_MARGIN)
                        index2, reason2, detail2 = S.take_back(sims, boxes, anchor, line=st.REGAIN_SAME, lead=st.REGAIN_MARGIN, leader_must_be_near=True)

                        def over(a_, b_):
                            """Two boxes' overlap, intersection over union."""
                            if a_ is None or b_ is None:
                                return None
                            ix, iy = min(a_[2], b_[2]) - max(a_[0], b_[0]), min(a_[3], b_[3]) - max(a_[1], b_[1])
                            inter = max(ix, 0) * max(iy, 0)
                            union = (a_[2] - a_[0]) * (a_[3] - a_[1]) + (b_[2] - b_[0]) * (b_[3] - b_[1]) - inter
                            return round(inter / union, 3) if union > 0 else 0.0
                        # every candidate over the line, with what a rule by place could use: so another rule can be
                        # read off this record without the card
                        candidates = [{"likeness": round(float(sims[i]), 4), "box": None if boxes[i] is None else [round(v, 3) for v in boxes[i]],
                                       "box_overlap_with_the_place_before_the_first_loss": over(boxes[i], anchor),
                                       "diagonals_from_it": None if boxes[i] is None or anchor is None else round(S._away(boxes[i], anchor)[0], 3)}
                                      for i in sorted(range(len(sims)), key=lambda k: -sims[k]) if sims[i] >= st.REGAIN_SAME]

                        def place(i):
                            """A taken candidate's box, and whether it is where the track is next found (boxes overlapping)."""
                            if i is None or boxes[i] is None:
                                return None
                            b, hit = boxes[i], None
                            if next_box is not None:
                                ix, iy = min(b[2], next_box[2]) - max(b[0], next_box[0]), min(b[3], next_box[3]) - max(b[1], next_box[1])
                                inter = max(ix, 0) * max(iy, 0)
                                union = (b[2] - b[0]) * (b[3] - b[1]) + (next_box[2] - next_box[0]) * (next_box[3] - next_box[1]) - inter
                                hit = round(inter / union, 3) if union > 0 else 0.0
                            away = None
                            if pick_box is not None:
                                away = round((((b[0] + b[2]) / 2 - (pick_box[0] + pick_box[2]) / 2) ** 2 + ((b[1] + b[3]) / 2 - (pick_box[1] + pick_box[3]) / 2) ** 2) ** 0.5, 3)
                            return {"box": [round(v, 3) for v in b], "box_overlap_with_the_tracks_next_mask": hit, "centre_from_the_picks_centre": away}
                        row["ways"][way] = {"detections": int(masks.shape[0]), "over_the_line": int(sum(s >= st.REGAIN_SAME for s in sims)),
                                            "best": round(float(best), 4), "next": round(float(second), 4), "lead": round(float(best - second), 4),
                                            "todays_rule_takes": place(taken), "take_back": {"reason": reason, "takes": place(index),
                                                                                             "distances_in_diagonals": detail.get("distances_in_diagonals")},
                                            "take_back_from_the_place_before_the_first_loss": {"reason": reason2, "takes": place(index2), "a_clear_leader_must_be_near": True,
                                                                                               "leader_distance_in_diagonals": detail2.get("leader_distance_in_diagonals"),
                                                                                               "distances_in_diagonals": detail2.get("distances_in_diagonals")},
                                            "over_the_line_candidates": candidates,
                                            **({"crop": crop} if crop else {})}
                    same = row["ways"]["16, the frame"]
                    row["reproduces_the_node"] = abs(same["best"] - row["the_node_then"]["best"]) < 1e-3 and abs(same["next"] - row["the_node_then"]["next"]) < 1e-3
                    rows.append(row)
                    print(name, json.dumps({"frame": row["frame"], "reproduces": row["reproduces_the_node"],
                                            **{w: [v["detections"], v["best"], v["lead"], bool(v["todays_rule_takes"]), v["take_back"]["reason"]] for w, v in row["ways"].items()}}), flush=True)
        picked = next((s_ for s_ in found.shots if s_.picked), None)
        seed = found.pieces.get(picked.start) if picked is not None else None
        track_summary = None
        fresh_looks = []
        if seed is not None:
            import torch.nn.functional as F
            on_ = seed > 0.5
            centres = {}
            for f_ in range(0, int(seed.shape[0]), 12):
                b_ = S.box_of(seed[f_])
                centres[int(f_ + picked.start)] = None if b_ is None else [round((b_[0] + b_[2]) / 2, 3), round((b_[1] + b_[3]) / 2, 3)]
            track_summary = {"frames": int(seed.shape[0]), "with_a_mask": int(on_.flatten(1).any(1).sum()),
                             "plausible": int(S.plausible(F.adaptive_max_pool2d(on_.float()[:, None], (252, 252))[:, 0] > 0.5).sum()),
                             "taken_again_at": [int(r[1]) for r in picked.regained], "box_centre_every_12_frames": centres}
            # DOES THE MASK STAY ON ONE FIGURE FROM FRAME TO FRAME? Each frame's mask against the frame before's, where
            # both have one: a person does not leave their own outline in a twenty-fourth of a second, so an overlap
            # near nothing is the mask moving to somebody else, with no empty frame to show it.
            steps = []
            for t_ in range(1, int(seed.shape[0])):
                now, was = on_[t_], on_[t_ - 1]
                if bool(now.any()) and bool(was.any()):
                    inter = int((now & was).sum())
                    steps.append((t_ + picked.start, inter / max(int(now.sum()) + int(was.sum()) - inter, 1)))

            def centre_of(f_):
                b_ = S.box_of(seed[f_ - picked.start])
                return None if b_ is None else [round((b_[0] + b_[2]) / 2, 3), round((b_[1] + b_[3]) / 2, 3)]
            low = [(f_, v_) for f_, v_ in steps if v_ < 0.5]
            if a.track_only:
                # every frame, for "one jump or a creep through neighbours": the box, and the overlap with the frame before's mask
                over_ = dict(steps)
                track_every_frame = [{"frame": int(f_), "box": (None if S.box_of(seed[f_ - picked.start]) is None else [round(v_, 3) for v_ in S.box_of(seed[f_ - picked.start])]),
                                      "overlap_with_the_frame_before": (None if f_ not in over_ else round(over_[f_], 3)),
                                      "share_of_the_frame": round(float(on_[f_ - picked.start].float().mean()), 5)}
                                     for f_ in range(picked.start, picked.start + int(seed.shape[0]))]
                # at each frame the node would look on, what a fresh detect returns and how many of its detections lie on the
                # finished track's mask there (mask against mask, `subject_tracks.AGREE_AT`): what a refresh would have to choose from
                for f_ in range(picked.start, picked.start + int(seed.shape[0]), st.PROBE_STRIDE):
                    fresh_, _ = detect16(f_)
                    m_ = on_[f_ - picked.start]
                    look_ = {"frame": int(f_), "asked": few, "detections": int(fresh_.shape[0]), "the_track_has_a_mask": bool(m_.any())}
                    if look_["the_track_has_a_mask"] and fresh_.shape[0]:
                        d_ = fresh_ > 0.5
                        inter_ = (d_ & m_).flatten(1).sum(1).float()
                        over_the_track = sorted((inter_ / (d_ | m_).flatten(1).sum(1).float().clamp(min=1)).tolist(), reverse=True)
                        look_.update({"on_the_track": sum(v_ >= S.AGREE_AT for v_ in over_the_track), "best_overlap": round(over_the_track[0], 3),
                                      "next_overlap": (round(over_the_track[1], 3) if len(over_the_track) > 1 else None)})
                    fresh_looks.append(look_)
            track_summary["mask_overlap_with_the_frame_before"] = {
                "pairs": len(steps), "least": None if not steps else round(min(v_ for _, v_ in steps), 3),
                "median": None if not steps else round(sorted(v_ for _, v_ in steps)[len(steps) // 2], 3),
                "under_a_half": [{"frame": int(f_), "overlap": round(v_, 3), "box_centre_before": centre_of(f_ - 1), "box_centre_after": centre_of(f_)} for f_, v_ in low]}
        # who each pick rule names on the pick frame, so a record can say whether two rules followed the same detection
        rules_ = {}
        if found.pick_frame is not None:
            on_pick, scores_pick = detect16(found.pick_frame)
            for rule_ in st.PICKS:
                i_ = st.choose(on_pick, scores_pick, rule_)
                rules_[rule_] = None if i_ is None or S.box_of(on_pick[i_]) is None else {"detection": int(i_), "box": [round(v_, 3) for v_ in S.box_of(on_pick[i_])]}
        R["runs"][name] = {"cuts": [int(c) for c in cuts], "pick_frame": found.pick_frame, "each_pick_rule_on_the_pick_frame": rules_,
                           "the_picked_subjects_box_on_the_pick_frame": (None if seed is None or found.pick_frame is None else
                                                                        [round(v, 3) for v in (S.box_of(seed[found.pick_frame - picked.start]) or [])]),
                           "the_track_handed_back": track_summary, "each_tracked_call": calls, "looks": rows,
                           **({"track_only": True, "the_track_every_frame": track_every_frame, "a_fresh_detect_at_each_look": fresh_looks} if a.track_only and seed is not None else {})}
        Path(a.json).write_text(json.dumps(R, indent=1))


def cmd_render(a):
    D = json.loads(Path(a.json).read_text())
    f = D["frames"]
    print(f"`{f['clip']}` from {f['second']} s, {f['count']} frames at {f['rate']} a second, {f['size'][0]}x{f['size'][1]}; SAM 3.1 {D['sam']}, on {D.get('device', 'the card')}; pick `{D['settings']['pick']}`; "
          f"the regain line {D['lines']['REGAIN_SAME']}, the lead required {D['lines']['REGAIN_MARGIN']}; the crop is {D['crop_times']} times the subject's last box, the centre region the middle "
          f"{D.get('centre_share')} of the frame in each direction. One run per arm.\n")

    def took(x):
        """A taken candidate as `yes (how far its centre is from the pick's centre, in frame units)`."""
        return "nobody" if x is None else f"yes ({x.get('centre_from_the_picks_centre')})"
    def box_overlap(a_, b_):
        ix, iy = min(a_[2], b_[2]) - max(a_[0], b_[0]), min(a_[3], b_[3]) - max(a_[1], b_[1])
        inter = max(ix, 0) * max(iy, 0)
        union = (a_[2] - a_[0]) * (a_[3] - a_[1]) + (b_[2] - b_[0]) * (b_[3] - b_[1]) - inter
        return inter / union if union > 0 else 0.0

    if any(run["looks"] for run in D["runs"].values()):
        print("### By place, read off the data\n\nFor each look and each count asked on the whole frame: the candidates over the likeness line, each as `likeness (overlap of its box with the picked "
              f"subject's box on the pick frame)`. `at the place` is the best of those overlapping by {AT_THE_PLACE} or more. This anchor is fair only where the subject stays where they stood.\n")
        print("| run | look, frame | asked | today's rule takes | candidates over the line | at the place |\n|---|---|---|---|---|---|")
        tally = {}
        for name, run in D["runs"].items():
            pick = run.get("the_picked_subjects_box_on_the_pick_frame")
            for row in run["looks"]:
                for way in ("16, the frame", "32, the frame", "64, the frame"):
                    v = row["ways"][way]
                    cands = [(c["likeness"], round(box_overlap(c["box"], pick), 3)) for c in v.get("over_the_line_candidates", []) if c["box"] and pick]
                    here = [c for c in cands if c[1] >= AT_THE_PLACE]
                    today = v["todays_rule_takes"]
                    took_ = "nobody" if today is None else f"somebody ({round(box_overlap(today['box'], pick), 3)})"
                    tally.setdefault((name, way), []).append((bool(here), today is not None and box_overlap(today["box"], pick) >= AT_THE_PLACE, today is not None))
                    print(f"| {name} | {row['frame']} | {way} | {took_} | {', '.join(f'{l} ({o})' for l, o in cands) or 'none'} | {max(here)[0] if here else 'nobody'} |")
        print("\n| run | asked | looks | looks with a candidate over the line at the place | today's rule takes somebody | of them at the place |\n|---|---|---|---|---|---|")
        for (name, way), rows_ in tally.items():
            print(f"| {name} | {way} | {len(rows_)} | {sum(r[0] for r in rows_)} | {sum(r[2] for r in rows_)} | {sum(r[1] for r in rows_)} |")
        print()
    for name, run in D["runs"].items():
        t = run.get("the_track_handed_back") or {}
        print(f"### {name}\n\nPick frame {run['pick_frame']}, the picked subject's box there {run.get('the_picked_subjects_box_on_the_pick_frame')} (left, top, right, bottom as shares of the frame), "
              f"cuts {run['cuts']}. The track the node hands back: {t.get('with_a_mask')} of {t.get('frames')} frames with a mask, {t.get('plausible')} plausible, taken again at {t.get('taken_again_at')}; "
              f"its box centre every twelve frames: {t.get('box_centre_every_12_frames')}. Each frame's mask against the frame before's: {t.get('mask_overlap_with_the_frame_before')}.\n")
        if run.get("each_tracked_call"):
            def step(x):
                return "none" if x is None else f"{x['frame']} (overlap {x['overlap']}, box ratio {x['box_ratio']}, centre step {x['centre_step']})"
            print(f"Each call the node made to the tracker, its own frames through `subject_tracks.unbroken` (a step under {run['each_tracked_call'][0]['moved_off']} cuts) "
                  "before any later take wrote over them. Frames are [first, one past the last).\n")
            print("| frames tracked | seeded on | unbroken | it stops, before and after | a gallery may use | least step inside | its last frame | the frame that stops it after |\n|---|---|---|---|---|---|---|---|")
            for c in run["each_tracked_call"]:
                print(f"| {c['frames']} | {c['seeded_on']} | {c['unbroken']} | {c['stops_before']}; {c['stops_after']} | {c['a_gallery_may_use']} | {c['least_overlap_inside']} | "
                      f"{step(c['its_last_frame'])} | {step(c['the_frame_that_stops_it_after'])} |")
            print()
        if run.get("the_track_every_frame"):
            rows_ = [x for x in run["the_track_every_frame"] if x.get("share_of_the_frame") is not None]
            first_ = next((x["share_of_the_frame"] for x in rows_ if x["frame"] == run["pick_frame"]), None)
            if first_:
                print("The mask's area against its area on the pick frame, every twelfth frame: "
                      + ", ".join(f"{x['frame']}: {x['share_of_the_frame'] / first_:.2f}" for x in rows_ if (x["frame"] - run["pick_frame"]) % 12 == 0) + ".\n")
        if run.get("a_fresh_detect_at_each_look"):
            print("A fresh detect on every frame the node would look on, against the finished track's mask there:\n\n"
                  "| frame | asked | detections | the track has a mask | detections on the track | best overlap | next |\n|---|---|---|---|---|---|---|")
            for x in run["a_fresh_detect_at_each_look"]:
                print(f"| {x['frame']} | {x['asked']} | {x['detections']} | {x['the_track_has_a_mask']} | {x.get('on_the_track', '')} | {x.get('best_overlap', '')} | {x.get('next_overlap', '')} |")
            print()
        if not run["looks"]:
            print("The node made no look after a loss: the track has no empty run to search.\n")
            continue
        print("The number after a `yes` is how far the taken candidate's centre is from the picked subject's centre on the pick frame, in frame units: on a stretch where the subject stays where "
              "they were, near is the subject and far is somebody else.\n")
        print("| look, frame | asked | detections | over the line | best | next | lead | today's rule takes | by place, from the node's last place | by place from the place before the first loss, a clear leader must be near | reproduces the node |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for row in run["looks"]:
            for way, v in row["ways"].items():
                first = v.get("take_back_from_the_place_before_the_first_loss")
                print(f"| {row['frame']} | {way} | {v['detections']} | {v['over_the_line']} | {v['best']} | {v['next']} | {v['lead']} | {took(v['todays_rule_takes'])} | "
                      f"{took(v['take_back']['takes']) if 'take_back' in v else 'withdrawn'} | {(took(first['takes']) + ': ' + first['reason']) if first else 'withdrawn'} | {row['reproduces_the_node'] if way == '16, the frame' else ''} |")
        print()


def _anchor(run: dict, track_run: dict | None, moved_off: float):
    """Where a re-find is anchored for one run: the subject's box on the last frame a gallery may use, and how it is known.

    From the run's own first tracked call when the json has it (`each_tracked_call`, judged on the masks). Otherwise from
    a `--track-only` json's every-frame list, by the same rule applied to its RECORDED steps: walking on from the pick
    frame, the run stops before the first frame with no mask or a step under the line, and the frame before a stop on
    a step is left out.
    """
    calls = run.get("each_tracked_call") or []
    if calls and calls[0].get("the_box_on_its_last_gallery_frame"):
        return tuple(calls[0]["the_box_on_its_last_gallery_frame"]), calls[0]["a_gallery_may_use"][1] - 1, "the pick's own tracked call, on its masks"
    if track_run is None:
        return None, None, "no anchor: the json has no tracked calls and no --track was given"
    rows = {x["frame"]: x for x in track_run["the_track_every_frame"]}
    last, moved = track_run["pick_frame"], False
    for f in range(track_run["pick_frame"] + 1, max(rows) + 1):
        x = rows.get(f)
        if x is None or x["box"] is None:
            break
        if x["overlap_with_the_frame_before"] is not None and x["overlap_with_the_frame_before"] < moved_off:
            moved = True
            break
        last = f
    if moved and last > track_run["pick_frame"]:
        last -= 1
    return tuple(rows[last]["box"]), last, "the recorded steps of a --track-only run"


def cmd_replay(a):
    """The recorded looks through `subject_tracks.take_back_by_place`, anchored where a gallery's last frame is."""
    sys.path.insert(0, str(REPO))
    import subject_tracks as S
    D = json.loads(Path(a.json).read_text())
    T = json.loads(Path(a.track).read_text()) if a.track else None
    line, lead = D["lines"]["REGAIN_SAME"], D["lines"]["REGAIN_MARGIN"]
    f = D["frames"]
    print(f"`{f['clip']}` from {f['second']} s, {f['count']} frames at {f['rate']} a second; pick `{D['settings']['pick']}`; SAM 3.1 {D['sam']}. Each recorded look through "
          f"`subject_tracks.take_back_by_place`: the line {line}, the lead {lead}, at the place from {S.AT_THE_PLACE} of box overlap. Nothing was detected again: the candidates are the "
          "recorded ones over the line, with the look's recorded runner-up added, without a box, where it is not among them.\n")
    print("| run | look, frame | asked | returned | cap reached | today's rule takes (overlap with the anchor) | by place, over the line | by place, first by a margin | "
          "the same anchored on the frame beside the jump |\n|---|---|---|---|---|---|---|---|---|")
    tally = {}
    for name, run in D["runs"].items():
        track_run = None if T is None else T["runs"].get(name)
        anchor, frame, how = _anchor(run, track_run, S.MOVED_OFF)
        beside = None if track_run is None or frame is None else next((tuple(x["box"]) for x in track_run["the_track_every_frame"] if x["frame"] == frame + 1 and x["box"]), None)
        print(f"<!-- {name}: anchored on frame {frame}, box {None if anchor is None else list(anchor)}, from {how}; the frame beside the jump: {None if beside is None else list(beside)} -->")
        for row in run["looks"]:
            for way, asked in (("16, the frame", 16), ("32, the frame", 32), ("64, the frame", 64)):
                v = row["ways"][way]
                cands = v.get("over_the_line_candidates", [])
                scores, boxes = [c["likeness"] for c in cands], [tuple(c["box"]) if c["box"] else None for c in cands]
                if v["next"] >= 0 and all(abs(v["next"] - s_) > 1e-4 for s_ in scores):
                    scores, boxes = scores + [v["next"]], boxes + [None]          # the runner-up, under the line: it decides "first, by a margin"

                def by(last_box, likeness):
                    i, why, _ = S.take_back_by_place(scores, boxes, last_box, asked=asked, line=line, lead=lead, likeness=likeness)
                    return (None, why) if i is None else (scores[i], why)
                over, first, at_beside = by(anchor, S.OVER_THE_LINE), by(anchor, S.FIRST_BY_A_MARGIN), by(beside, S.OVER_THE_LINE)
                today = v["todays_rule_takes"]
                today_at = None if today is None else round(S.box_overlap(tuple(today["box"]), anchor), 3)
                tally.setdefault((name, way), []).append((today is not None, today_at is not None and today_at >= S.AT_THE_PLACE, over[0] is not None, first[0] is not None, at_beside[0] is not None))
                print(f"| {name} | {row['frame']} | {asked} | {v['detections']} | {v['detections'] >= asked} | {'nobody' if today is None else f'somebody ({today_at})'} | "
                      f"{over[0] if over[0] is not None else 'nobody: ' + over[1]} | {first[0] if first[0] is not None else 'nobody'} | {at_beside[0] if at_beside[0] is not None else 'nobody'} |")
    print("\n| run | asked | looks | today's rule takes somebody | of them at the place | by place, over the line, takes | by place, first by a margin, takes | anchored on the frame beside the jump, takes |\n|---|---|---|---|---|---|---|---|")
    for (name, way), rows_ in tally.items():
        print(f"| {name} | {way} | {len(rows_)} | {sum(r[0] for r in rows_)} | {sum(r[1] for r in rows_)} | {sum(r[2] for r in rows_)} | {sum(r[3] for r in rows_)} | {sum(r[4] for r in rows_)} |")


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.set_defaults(fn=cmd_run)
    s.add_argument("--clip", required=True)
    s.add_argument("--second", type=float, required=True)
    s.add_argument("--seconds", type=float, required=True)
    s.add_argument("--width", type=int, required=True, help="the loader's custom_width")
    s.add_argument("--rate", type=float, required=True, help="the loader's force_rate")
    s.add_argument("--json", required=True)
    s.add_argument("--corrected", action="store_true", help="run SAM 3.1 through sam31_corrections; today's node runs it as shipped")
    s.add_argument("--track-only", action="store_true", help="follow and record the track handed back; ask nothing again at the looks")
    s.add_argument("--pick", required=True, choices=["largest", "most central", "best match for the phrase"],
                   help="the node's `pick`. Always named and never read from h3_config: the shipped default changed on 2026-10-07, and a record must say who it followed")
    s = sub.add_parser("render")
    s.set_defaults(fn=cmd_render)
    s.add_argument("--json", required=True)
    s = sub.add_parser("replay", help="the recorded looks through subject_tracks.take_back_by_place; no model, no card")
    s.set_defaults(fn=cmd_replay)
    s.add_argument("--json", required=True, help="a `run` json with its looks")
    s.add_argument("--track", help="a `run --track-only` json of the same window and pick, for the anchor when --json has no tracked calls")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
