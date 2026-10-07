#!/usr/bin/env python3
"""Is ONE subject in a crowd followed better alone, or seeded together with the people around them?

Today's Subject Track seeds ComfyUI's tracker with its one subject (`subject_track.py::_sam_callables`, `max_objects=1`).
On 2026-10-07 a subject followed alone on a dense frame was lost within a few frames, in ComfyUI's port and in Meta's own
tracker alike, while sixteen subjects seeded together mostly held (`bench/results/2026-10-07_sam3_core_against_meta.md`,
the there-and-back rung); on another clip company made no such difference. The tracker node to come has to choose, so
this measures it on the footage the choice is for: the same subject, the same seed frame, followed

    alone                         as today's node does
    with the three nearest        a little company
    in a group of up to sixteen   the most prominent people on the seed frame, the subject among them

each with ComfyUI's SAM 3.1 as it ships and with `sam31_corrections.corrected`, and each again with every input value
moved one level of 255, so a difference between arms is read against what that does to each. What is counted is the
SUBJECT's track only: frames with a mask, frames whose mask passes the shape test, frames that are trusted (shape, and
not contradicted by the detector on the looked-at frames; `subject_tracks.py`), and the first frame it is lost on.

    <python> bench/subject_alone_or_in_a_group.py run --clip C --second S --seconds T --width W --rate R --subject P --json J [--key K]
    <python> bench/subject_alone_or_in_a_group.py render --json J

THE READING, WRITTEN BEFORE THE FIRST RUN (2026-10-07). "In a group" is the tracker node's design only if, on the hard
crowd stretch, the subject's trusted frames in a group exceed alone by more than the larger of the two arms' own
nudged-against-plain differences, in BOTH the plain and the nudged pair, and it is not worse by that much on the calm
stretches. Two limits written with it: "trusted" cannot see a track that swapped onto a detected neighbour, so each group
arm also reports its overlap with the alone arm where both have a mask (a low one means one of the two is on somebody
else); and "as ComfyUI ships it" against "corrected" changes the detector behind the looks as well as the tracker, so
only alone against group WITHIN one of them is a clean comparison. If neither is clearly better, the node follows alone, as today, since alone costs least. Trusted frames are
agreement between track and detector, not proof of identity; the owner's eye on the stacked masks is the last word.

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

#: reasoned: the masks are compared on a quarter of the model's grid; `subject_tracks.plausible` works in shares of the frame.
SMALL = 252
#: inherited: `subject_track.PROBE_OFFSET` and `PROBE_STRIDE`, the frame today's node judges a shot on and how often it looks.
SEED_FRAME = 4
LOOK_EVERY = 12
#: reasoned, not measured: a mask that shares less than this with the frame before's, by intersection over union, has
#: moved off the figure it was on; a person keeps most of their outline over a twenty-fourth of a second.
MOVED_OFF = 0.2
#: inherited: `h3_config.SUBJECT_TRACK["max_people"]`, the most people the detector is asked for; also one group of the tracker.
MOST = 16
COMPANY = (("alone", 1), ("with the three nearest", 4), ("in a group of up to sixteen", MOST))
#: inherited: core's own line for "the same object" between detections (`comfy/ldm/sam3/tracker.py`, its mask overlap removal).
SAME_OBJECT = 0.5


def cmd_run(a):
    torch = boot(float32=False)
    import torch.nn.functional as F

    import server
    server.PromptServer.instance = types.SimpleNamespace(prompt_queue=None, routes=None)
    sys.path.insert(0, str(COMFY / "custom_nodes" / "ComfyUI-VideoHelperSuite"))
    from videohelpersuite.load_video_nodes import LoadVideoFFmpegPath
    from comfy.ldm.sam3.tracker import unpack_masks
    from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_VideoTrack

    pkg = types.ModuleType("_h3pack")       # the node's module imports relatively; give it a package to live in
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    import _h3pack.subject_track as st

    import sam31_corrections
    import subject_tracks
    from h3_config import SEGMENTER

    loaded = LoadVideoFFmpegPath().load_video(video=a.clip, force_rate=float(a.rate), custom_width=a.width, custom_height=0,
                                              frame_load_cap=int(round(a.seconds * a.rate)), start_time=float(a.second), format="AnimateDiff")[0][..., :3]
    u8 = (loaded.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    del loaded
    n, H, W = (int(v) for v in u8.shape[:3])
    stock, stock_clip = load_core(SEGMENTER)
    fixed, fixed_clip = sam31_corrections.corrected(stock, stock_clip)
    R = {"environment": environment(torch), "frames": {"clip": Path(a.clip).name, "second": a.second, "count": n, "rate": a.rate, "size": [W, H]},
         "seed_frame": SEED_FRAME, "look_every": LOOK_EVERY, "one_run_per_arm": True, "arms": {}}

    def small(x):
        if x.shape[0] == 0:      # a look that found nobody
            return torch.zeros((0, SMALL, SMALL), dtype=torch.bool)
        return F.adaptive_max_pool2d(x.float()[:, None], (SMALL, SMALL))[:, 0] > 0.5

    def detector(model, clip, video):
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(f"person:{MOST}"))

        def detect(f):
            out = SAM3_Detect.execute(model, video[f:f + 1], conditioning=cond, threshold=0.5, individual_masks=True)
            return getattr(out, "args", out)[0].float().cpu()
        return detect

    # ONE subject and ONE set of seeds for every arm: found once, on the frames as fed, by SAM as it ships. A nudge or
    # a correction can change who is largest on the seed frame, and the arms would then follow different people.
    plain = torch.from_numpy(u8.astype(np.float32) / 255.0)
    with torch.inference_mode():
        found = detector(stock, stock_clip, plain)(SEED_FRAME)
    if found.shape[0] == 0:
        raise SystemExit("nothing detected on the seed frame")
    area = found.flatten(1).sum(1)
    # the subject by the node's own rule (`subject_track.choose`). The first runs of this tool took the largest, the
    # shipped `pick`, and on all three windows that is a figure low in the frame or cut by its edge, not the central one
    subject = int(st.choose(found, [], a.subject))
    box = subject_tracks.box_of(found[subject])
    R["subject"] = {"picked_by": a.subject, "box_on_the_seed_frame": None if box is None else [round(v, 3) for v in box]}
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    cx, cy = (found * xs).flatten(1).sum(1) / area.clamp(min=1), (found * ys).flatten(1).sum(1) / area.clamp(min=1)
    by_distance = ((cx - cx[subject]) ** 2 + (cy - cy[subject]) ** 2).argsort().tolist()
    nearest = [subject] + [i for i in by_distance if i != subject]                               # the subject first, then by distance
    largest = [subject] + [i for i in area.argsort(descending=True).tolist() if i != subject]   # the subject first, then by size

    flat = (found > 0.5).flatten(1).float()

    def overlap_of(i, j) -> float:
        """The larger of intersection over union and intersection over the smaller mask: core's own measure of "the same object"."""
        inter = float((flat[i] * flat[j]).sum())
        return max(inter / max(float(flat[i].sum() + flat[j].sum()) - inter, 1.0), inter / max(min(float(flat[i].sum()), float(flat[j].sum())), 1.0))

    def distinct(order, k):
        """The first `k` of `order` that are not a second mask of somebody already chosen. Core's detect node removes no
        overlapping detections, so two masks of one person can come back; seeding both would set core's own rules between
        objects against the subject, which is not "company"."""
        chosen, dropped = [], 0
        for i in order:
            if any(overlap_of(i, j) >= SAME_OBJECT for j in chosen):
                dropped += 1
                continue
            chosen.append(i)
            if len(chosen) == k:
                break
        return chosen, dropped

    SEEDS, R["seed_sets"] = {}, {}
    for name, k in COMPANY:
        chosen, dropped = distinct(largest if k == MOST else nearest, k)
        SEEDS[k] = found[chosen]
        R["seed_sets"][name] = {"seeded": len(chosen), "second_masks_of_a_chosen_person_dropped": dropped,
                                "largest_overlap_with_the_subject": round(max([overlap_of(subject, j) for j in chosen[1:]] or [0.0]), 3)}
    R["detections_on_the_seed_frame"] = int(found.shape[0])
    R["seeds"] = "one detect on the seed frame as fed, by SAM as it ships; the same masks seed every arm, the subject's first"

    def run(model, clip, levels: int, k: int, every_track: bool = False):
        """The subject's track from the seed frame to the end, [F, SMALL, SMALL] bool, with what the detector saw on the looks.

        `every_track` also counts where two of the followed tracks lie on one figure, and leaves out the looks.
        """
        video = plain if levels == 0 else torch.from_numpy(nudged(u8, levels).astype(np.float32) / 255.0)
        detect = detector(model, clip, video)
        seeds = SEEDS[k]
        merged, every = None, None
        with torch.inference_mode():
            out = SAM3_VideoTrack.execute(video[SEED_FRAME:], model, initial_mask=seeds, conditioning=None, detection_threshold=0.5, max_objects=0, detect_interval=1)
            packed = getattr(out, "args", out)[0]["packed_masks"]
            track = torch.zeros((n - SEED_FRAME, SMALL, SMALL), dtype=torch.bool)
            if packed is not None:
                track = small(unpack_masks(packed[:, 0]).cpu())      # row 0: the subject's seed is first in every set
            if every_track and packed is not None:
                rows = torch.stack([small(unpack_masks(packed[:, j]).cpu()) for j in range(int(seeds.shape[0]))]).flatten(2).float()     # [K, F, P]
                every = rows > 0.5
                pairs, frames_, with_subject = set(), set(), set()
                for t in range(rows.shape[1]):
                    x = rows[:, t]
                    inter = x @ x.T
                    area = x.sum(1)
                    iou = inter / (area[:, None] + area[None, :] - inter).clamp(min=1)
                    for i, j in torch.triu(iou > SAME_OBJECT, diagonal=1).nonzero().tolist():
                        pairs.add((i, j))
                        frames_.add(t)
                        if i == 0:
                            with_subject.add(t)
                merged = {"pairs_of_tracks_sharing_over_a_half_of_their_masks_on_some_frame": len(pairs), "frames_with_such_a_pair": len(frames_),
                          "frames_where_the_subject_is_in_one": len(with_subject)}
            looks = {} if every_track else {f - SEED_FRAME: small(detect(f)) for f in range(SEED_FRAME, n, LOOK_EVERY)}
        on = track.flatten(1).any(1)
        ok = subject_tracks.plausible(track)
        trust = subject_tracks.trusted(track, looks)
        lost = (~on).nonzero()
        # DOES THE MASK STAY ON ONE FIGURE? Each frame's mask against the frame before's. Added 2026-10-07 after a track
        # followed alone was found to move to another figure half the frame away with no empty frame between: a count
        # of plausible masks cannot see that, and "held" below is the count that can.
        steps = {}
        for t in range(1, int(track.shape[0])):
            if bool(on[t]) and bool(on[t - 1]):
                steps[t] = int((track[t] & track[t - 1]).sum()) / max(int((track[t] | track[t - 1]).sum()), 1)
        # held: up to the first frame that is empty or has moved off. NOT "or is not plausible": the first run with
        # this count (the central figure, hard stretch) had a mask on every frame, every step over 0.6, and one or two
        # frames the shape test failed, and "held" then read 140 in one arm and 89 in its nudged twin. A single frame
        # the shape test fails on a moving figure is a false alarm of that test, counted beside it, not a break.
        held = 0
        for t in range(int(track.shape[0])):
            if not bool(on[t]) or steps.get(t, 1.0) < MOVED_OFF:
                break
            held += 1
        not_plausible = [int(t) + SEED_FRAME for t in (on & ~ok).nonzero().flatten().tolist()]
        values = sorted(steps.values())
        return (track if every is None else every), {"seeded": int(seeds.shape[0]), "frames": int(on.numel()),
                       "with_a_mask": int(on.sum()), "plausible": int(ok.sum()), "trusted": int(trust.sum()),
                       "first_frame_lost": (int(lost[0]) + SEED_FRAME) if lost.numel() else None,
                       "doubted": int(subject_tracks.doubted(track, looks).sum()),
                       "held_from_the_seed_without_a_break": held, "held_counts": "up to the first frame that is empty or has moved off",
                       "frames_with_a_mask_the_shape_test_fails": not_plausible[:20], **({"two_tracks_on_one_figure": merged} if merged is not None else {}),
                       "mask_overlap_with_the_frame_before": {"least": round(values[0], 3) if values else None, "median": round(values[len(values) // 2], 3) if values else None,
                                                             "under_a_half": [[t + SEED_FRAME, round(v, 3)] for t, v in steps.items() if v < 0.5]}}

    def overlap(x, y):
        inter, union = (x & y).flatten(1).sum(1).float(), (x | y).flatten(1).sum(1).float()
        on = union > 0
        return round(float((inter[on] / union[on]).mean()), 3) if bool(on.any()) else None

    def overlap_where_both(x, y):
        both = x.flatten(1).any(1) & y.flatten(1).any(1)
        if not bool(both.any()):
            return None
        inter, union = (x[both] & y[both]).flatten(1).sum(1).float(), (x[both] | y[both]).flatten(1).sum(1).float().clamp(min=1)
        return {"frames_both_have_a_mask": int(both.sum()), "mean": round(float((inter / union).mean()), 3), "frames_under_a_half": int(((inter / union) < 0.5).sum())}

    def save():
        f = Path(a.json)
        data = json.loads(f.read_text()) if f.exists() else {}
        data[a.key or f"{R['frames']['clip']} from {a.second} s"] = R
        f.write_text(json.dumps(data, indent=1))

    if a.rules:
        # CORE'S TWO RULES BETWEEN FOLLOWED OBJECTS, each switched off in memory for the length of one arm; nothing on
        # disk is touched. First read in `docs/research/masking/2026-10-07_mryolk_stage_table.md`, row 26, with lines
        # for both sides; this run is what tests it. No arm here is Meta's rule whole: Meta's blanks the returned mask
        # as well as the stored one, and its session logic around it differs (the same table, row 26b: not tested). (1) `_suppress_recently_occluded`: of two tracks whose masks overlap by 0.3 (the larger of
        # intersection over union and over the smaller mask), the one more recently empty is blanked. Meta's full
        # pipeline has the same rule at 0.7 of intersection over union alone (`meta_sam3`, `model_builder.py`,
        # `suppress_overlapping_based_on_recent_occlusion_threshold`; `sam3_video_base.py`). (2) in
        # `_deferred_memory_encode`: a track that keeps under 0.3 of its area once every pixel is given to one track
        # gets an empty memory for that frame. Meta's has the same rule at the same 0.3.
        import functools
        import inspect
        import textwrap
        import types as types_
        import comfy.ldm.sam3.tracker as core_tracker

        def tracker_of(model):
            return next(m for m in model.model.diffusion_model.modules() if hasattr(m, "_suppress_recently_occluded") and hasattr(m, "_deferred_memory_encode"))

        def no_shrink(obj):
            src = textwrap.dedent(inspect.getsource(type(obj)._deferred_memory_encode))
            line = "shrink_ok = (area_after / area_before) >= 0.3"
            if src.count(line) != 1:
                raise SystemExit("core's memory rule no longer reads as this tool expects; read comfy/ldm/sam3/tracker.py")
            scope: dict = {}
            exec(src.replace(line, "shrink_ok = (area_after / area_before) >= 0.0"), vars(core_tracker), scope)
            return types_.MethodType(scope["_deferred_memory_encode"], obj)

        def unsuppressed(masks, last_occluded, frame_idx, threshold=0.3):
            return masks

        RULES = {"as core has them": {}, "the occlusion rule off": {"_suppress_recently_occluded": lambda o: unsuppressed},
                 "the occlusion rule at 0.7, core's measure": {"_suppress_recently_occluded": lambda o: functools.partial(type(o)._suppress_recently_occluded, threshold=0.7)},
                 "the occlusion rule at 0.7 of intersection over union (Meta's threshold and measure)": {"_suppress_recently_occluded": lambda o: union_alone_rule(type(o))},
                 "the memory rule off": {"_deferred_memory_encode": no_shrink},
                 "both off": {"_suppress_recently_occluded": lambda o: unsuppressed, "_deferred_memory_encode": no_shrink}}
        R["rules"] = "each rule switched off in memory on core's tracker for one arm; the subject in the group of up to sixteen"
        # THE PREDICTION, WRITTEN BEFORE THE FIRST RUN (2026-10-07, from the code): the occlusion rule blanks one of an
        # overlapping pair only when BOTH have been empty or blanked before (`comfy/ldm/sam3/tracker.py`, `last_occ_j >
        # -1` beside `last_occ_i > last_occ_j`; Meta's has the same condition, `meta_sam3/sam3/model/sam3_video_base.py`).
        # So every arm that changes only that rule must give the same masks as core as it is, on every track, up to the
        # first frame on which any track is empty. If an arm differs earlier, the patch is doing something else and
        # the arm is void. Compared on the masks pooled to the tool's small grid.
        R["prediction"] = ("an arm that changes only the occlusion rule equals core as it is on every track up to the first frame on which any track is empty; "
                           "an earlier difference voids the arm")
        base = {}
        for model_name, model, clip in (("as ComfyUI ships it", stock, stock_clip), ("corrected", fixed, fixed_clip)):
            obj = tracker_of(model)
            for rule, patches in RULES.items():
                for name_, make in patches.items():
                    setattr(obj, name_, make(obj))
                try:
                    for levels in (0, 1):
                        key = f"{model_name} | {rule}" + (" | nudged" if levels else "")
                        every, R["arms"][key] = run(model, clip, levels, MOST, every_track=True)
                        if not patches:
                            base[(model_name, levels)] = every
                            empty = (~every.any(2)).any(0).nonzero()             # frames on which some track has no mask
                            R["arms"][key]["first_frame_on_which_any_track_is_empty"] = (int(empty[0]) + SEED_FRAME) if empty.numel() else None
                        elif (model_name, levels) in base and base[(model_name, levels)].shape == every.shape:
                            differs = (every != base[(model_name, levels)]).any(2).any(0).nonzero()
                            first_empty = R["arms"][f"{model_name} | as core has them" + (" | nudged" if levels else "")]["first_frame_on_which_any_track_is_empty"]
                            first_diff = (int(differs[0]) + SEED_FRAME) if differs.numel() else None
                            R["arms"][key]["first_frame_differing_from_core_as_it_is"] = first_diff
                            if "_deferred_memory_encode" not in patches:
                                R["arms"][key]["the_prediction_holds"] = first_diff is None or (first_empty is not None and first_diff >= first_empty)
                        save()
                        print(key, json.dumps(R["arms"][key]), flush=True)
                finally:
                    for name_ in patches:
                        delattr(obj, name_)
                torch.cuda.empty_cache()
        save()
        return

    for model_name, model, clip in (("as ComfyUI ships it", stock, stock_clip), ("corrected", fixed, fixed_clip)):
        alone = {}                                           # the alone arm's track, per nudge: what each group arm is held against
        for company, k in COMPANY:
            tracks = {}
            for levels in (0, 1):
                key = f"{model_name} | {company}" + (" | nudged" if levels else "")
                tracks[levels], R["arms"][key] = run(model, clip, levels, k)
                if k == 1:
                    alone[levels] = tracks[levels]
                elif levels in alone:
                    # a low overlap while both have a mask means the two arms are on different people: one of them swapped
                    R["arms"][key]["overlap_with_the_alone_arm"] = overlap_where_both(tracks[levels], alone[levels])
                save()
                print(key, json.dumps(R["arms"][key]), flush=True)
            if 0 in tracks and 1 in tracks:
                R["arms"][f"{model_name} | {company}"]["against_itself_nudged"] = overlap(tracks[0], tracks[1])
            torch.cuda.empty_cache()
    save()


def union_alone_rule(cls):
    """Core's occlusion rule with Meta's measure and threshold: 0.7 of intersection over union alone. Still applied
    where core applies it, to the stored mask; Meta's also blanks the returned one (the stage table, row 26)."""
    import functools
    import inspect
    import textwrap

    import comfy.ldm.sam3.tracker as core_tracker
    src = textwrap.dedent(inspect.getsource(cls._suppress_recently_occluded))
    line = "iou = _compute_mask_overlap(low_res_masks[:, 0], low_res_masks[:, 0])"
    if src.count(line) != 1 or not src.startswith("@staticmethod"):
        raise SystemExit("core's occlusion rule no longer reads as this tool expects; read comfy/ldm/sam3/tracker.py")
    new = ("flat_ = binary.float().flatten(1); inter_ = flat_ @ flat_.T; area_ = flat_.sum(1, keepdim=True); "
           "iou = inter_ / (area_ + area_.T - inter_).clamp(min=1)")
    scope: dict = {}
    exec(src.replace("@staticmethod\n", "", 1).replace(line, new), vars(core_tracker), scope)
    return functools.partial(scope["_suppress_recently_occluded"], threshold=0.7)


def cmd_rules_selftest(a):
    """Model-free, off the card: do the two forms of the occlusion rule part where the reading says they do?"""
    import torch

    sys.path.insert(0, str(COMFY))
    argv, sys.argv = sys.argv, sys.argv[:1]         # ComfyUI reads the command line when its arguments are imported
    import comfy.cli_args
    comfy.cli_args.args.cpu = True                  # no model is loaded and the card is not touched
    sys.argv = argv
    import comfy.ldm.sam3.tracker as core_tracker
    cls = next(c for c in vars(core_tracker).values() if isinstance(c, type) and hasattr(c, "_suppress_recently_occluded") and hasattr(c, "_deferred_memory_encode"))
    union = union_alone_rule(cls)

    def blanked(rule, masks):
        out = masks.clone()
        rule(out, torch.tensor([3, 5]), 9)          # the second object was empty more recently: it is the one a rule would blank
        return bool((out[1] > 0).sum() == 0)
    inside = torch.full((2, 1, 8, 8), -5.0)         # a small mask lying inside a larger one: over the smaller 1.0, over the union 0.25
    inside[0, 0, :4, :4], inside[1, 0, :2, :2] = 5, 5
    same = torch.full((2, 1, 8, 8), -5.0)           # THE CONTROL: two masks that are nearly one: over the union 0.8
    same[0, 0, :4, :5], same[1, 0, :4, :4] = 5, 5
    apart = torch.full((2, 1, 8, 8), -5.0)          # and two that do not touch
    apart[0, 0, :2, :2], apart[1, 0, 5:, 5:] = 5, 5
    got = {"a small mask inside a larger one": (blanked(cls._suppress_recently_occluded, inside), blanked(union, inside)),
           "two masks that are nearly one": (blanked(cls._suppress_recently_occluded, same), blanked(union, same)),
           "two masks apart": (blanked(cls._suppress_recently_occluded, apart), blanked(union, apart))}
    want = {"a small mask inside a larger one": (True, False), "two masks that are nearly one": (True, True), "two masks apart": (False, False)}
    for k in want:
        print(("ok   " if got[k] == want[k] else "FAIL ") + f"{k}: core's rule blanks the more recently empty one: {got[k][0]}; at 0.7 of intersection over union: {got[k][1]}")
    return 0 if got == want else 1


def held(r: dict, seed_frame: int):
    """Frames from the seed up to the first that is empty or has moved off, read from an arm's record."""
    if "mask_overlap_with_the_frame_before" not in r:
        return None
    ends = [f for f, v in r["mask_overlap_with_the_frame_before"]["under_a_half"] if v < MOVED_OFF]
    if r.get("first_frame_lost") is not None:
        ends.append(r["first_frame_lost"])
    return (min(ends) - seed_frame) if ends else r["frames"]


def cmd_render(a):
    for name, D in json.loads(Path(a.json).read_text()).items():
        f = D["frames"]
        if D.get("rules"):
            print(f"### {name}\n\n`{f['clip']}` from {f['second']} s, {f['count']} frames at {f['rate']} a second, {f['size'][0]}x{f['size'][1]}; the subject is the {D['subject']['picked_by']} person on frame "
                  f"{D['seed_frame']}, box {D['subject']['box_on_the_seed_frame']}, seeded with the group of up to sixteen; {D['rules']}. One run per arm.\n")
            print("| SAM 3.1 | core's rules between followed objects | the subject held from the seed without a break | nudged | with a mask | nudged | pairs of tracks sharing over a half of their masks on some frame | "
                  "frames with such a pair | of them with the subject | nudged: pairs, frames, with the subject |\n|---|---|---|---|---|---|---|---|---|---|")
            for key, r in D["arms"].items():
                if key.endswith("| nudged"):
                    continue
                model, rule = key.split(" | ")
                t = D["arms"].get(key + " | nudged", {})
                m, tm = r["two_tracks_on_one_figure"], t.get("two_tracks_on_one_figure", {})
                print(f"| {model} | {rule} | {held(r, D['seed_frame'])} | {held(t, D['seed_frame']) if t else None} | {r['with_a_mask']} | {t.get('with_a_mask')} | {list(m.values())[0]} | {list(m.values())[1]} | {list(m.values())[2]} | {list(tm.values())} |")
            print(f"\nThe prediction written before the run: {D.get('prediction')}.\n")
            print("| SAM 3.1 | core's rules between followed objects | first frame on which any track is empty (core as it is) | first frame differing from core as it is | nudged | the prediction holds | nudged |\n|---|---|---|---|---|---|---|")
            for key, r in D["arms"].items():
                if key.endswith("| nudged"):
                    continue
                model, rule = key.split(" | ")
                t = D["arms"].get(key + " | nudged", {})
                print(f"| {model} | {rule} | {r.get('first_frame_on_which_any_track_is_empty', '')} | {r.get('first_frame_differing_from_core_as_it_is', '')} | {t.get('first_frame_differing_from_core_as_it_is', '')} | "
                      f"{r.get('the_prediction_holds', '')} | {t.get('the_prediction_holds', '')} |")
            print()
            continue
        print(f"### {name}\n\n`{f['clip']}` from {f['second']} s, {f['count']} frames at {f['rate']} a second, {f['size'][0]}x{f['size'][1]}; the subject is the {D.get('subject', {}).get('picked_by', 'largest')} person "
              f"on frame {D['seed_frame']}{', box ' + str(D['subject']['box_on_the_seed_frame']) + ' (left, top, right, bottom as shares of the frame)' if D.get('subject') else ''} ({D.get('detections_on_the_seed_frame')} detections; {D.get('seeds', '')}), followed from there; the detector is asked again every {D['look_every']} frames. One run per arm.\n")
        print(f"Seed sets: {D.get('seed_sets')}.\n")
        print("| SAM 3.1 | the subject is seeded | seeded | frames | with a mask | plausible | trusted | first frame lost | the same, nudged: with a mask, plausible, trusted | overlap with its nudged twin | against the alone arm where both have a mask: frames, mean overlap, frames under a half |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for key, r in D["arms"].items():
            if key.endswith("| nudged") or "error" in r:
                continue
            model, company = key.split(" | ")
            t = D["arms"].get(key + " | nudged", {})
            print(f"| {model} | {company} | {r['seeded']} | {r['frames']} | {r['with_a_mask']} | {r['plausible']} | {r['trusted']} | {r['first_frame_lost']} | "
                  f"{t.get('with_a_mask')}, {t.get('plausible')}, {t.get('trusted')} | {r.get('against_itself_nudged')} | "
                  f"{'' if not r.get('overlap_with_the_alone_arm') else list(r['overlap_with_the_alone_arm'].values())} |")
        print()
        if any("mask_overlap_with_the_frame_before" in r for r in D["arms"].values()):
            print("Does the mask stay on one figure? `held` counts frames from the seed frame up to the first that is empty or shares under "
                  f"{MOVED_OFF} of its mask with the frame before's.\n")
            print("| SAM 3.1 | the subject is seeded | held from the seed without a break | nudged | each mask against the frame before's: least, median | frames under a half (frame, overlap) | nudged: least, frames under a half |\n|---|---|---|---|---|---|---|")
            for key, r in D["arms"].items():
                if key.endswith("| nudged") or "error" in r or "mask_overlap_with_the_frame_before" not in r:
                    continue
                model, company = key.split(" | ")
                t, m = D["arms"].get(key + " | nudged", {}), r["mask_overlap_with_the_frame_before"]
                tm = t.get("mask_overlap_with_the_frame_before", {})
                print(f"| {model} | {company} | {held(r, D['seed_frame'])} | {held(t, D['seed_frame'])} | {m['least']}, {m['median']} | {m['under_a_half']} | {tm.get('least')}, {tm.get('under_a_half')} |")
            print()


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
    s.add_argument("--key", default="", help="the name of this run in the json; the clip and second when left out")
    s.add_argument("--rules", action="store_true", help="instead of the company arms: the group of up to sixteen with core's two rules between followed objects switched off in memory, one at a time and both")
    s.add_argument("--subject", required=True, choices=["largest", "most central"],
                   help="the node's `pick` rule that chooses the subject on the seed frame. Always named: the first runs took the largest, the shipped default until 2026-10-07")
    s = sub.add_parser("rules-selftest")
    s.set_defaults(fn=cmd_rules_selftest)
    s = sub.add_parser("render")
    s.set_defaults(fn=cmd_render)
    s.add_argument("--json", required=True)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
