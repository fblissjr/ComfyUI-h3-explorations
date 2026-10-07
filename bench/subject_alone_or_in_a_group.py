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

    <python> bench/subject_alone_or_in_a_group.py run --clip C --second S --seconds T --width W --rate R --json J [--key K]
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
    subject = int(area.argmax())
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

    def run(model, clip, levels: int, k: int):
        """The subject's track from the seed frame to the end, [F, SMALL, SMALL] bool, with what the detector saw on the looks."""
        video = plain if levels == 0 else torch.from_numpy(nudged(u8, levels).astype(np.float32) / 255.0)
        detect = detector(model, clip, video)
        seeds = SEEDS[k]
        with torch.inference_mode():
            out = SAM3_VideoTrack.execute(video[SEED_FRAME:], model, initial_mask=seeds, conditioning=None, detection_threshold=0.5, max_objects=0, detect_interval=1)
            packed = getattr(out, "args", out)[0]["packed_masks"]
            track = torch.zeros((n - SEED_FRAME, SMALL, SMALL), dtype=torch.bool)
            if packed is not None:
                track = small(unpack_masks(packed[:, 0]).cpu())      # row 0: the subject's seed is first in every set
            looks = {f - SEED_FRAME: small(detect(f)) for f in range(SEED_FRAME, n, LOOK_EVERY)}
        on = track.flatten(1).any(1)
        ok = subject_tracks.plausible(track)
        trust = subject_tracks.trusted(track, looks)
        lost = (~on).nonzero()
        return track, {"seeded": int(seeds.shape[0]), "frames": int(on.numel()),
                       "with_a_mask": int(on.sum()), "plausible": int(ok.sum()), "trusted": int(trust.sum()),
                       "first_frame_lost": (int(lost[0]) + SEED_FRAME) if lost.numel() else None,
                       "doubted": int(subject_tracks.doubted(track, looks).sum())}

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


def cmd_render(a):
    for name, D in json.loads(Path(a.json).read_text()).items():
        f = D["frames"]
        print(f"### {name}\n\n`{f['clip']}` from {f['second']} s, {f['count']} frames at {f['rate']} a second, {f['size'][0]}x{f['size'][1]}; the subject is the largest person "
              f"on frame {D['seed_frame']} ({D.get('detections_on_the_seed_frame')} detections; {D.get('seeds', '')}), followed from there; the detector is asked again every {D['look_every']} frames. One run per arm.\n")
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
    s = sub.add_parser("render")
    s.set_defaults(fn=cmd_render)
    s.add_argument("--json", required=True)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
