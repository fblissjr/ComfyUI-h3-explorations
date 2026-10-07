#!/usr/bin/env python3
"""Does the Subject Track decide each shot the same way when every input pixel moves by one level of 255?

The Subject Track (`subject_track.py`) finds cuts, picks a subject, and in every other shot takes the person most like
them if the likeness reaches a line; inside a shot it seeds the track again after a loss when one person is clearly the
subject. Those lines have a few hundredths of margin and were set on one clip. On 2026-10-07 a one-level change of the
input was measured to move which masks the detector returns on crowded frames and to move tracked masks on hard footage
(`bench/results/2026-10-07_sam3_precision_arms.md`). This tool asks whether the decisions between shots survive that
change: the node's own functions (`cut_scores`, `auto_cuts`, `find_cuts`, `_sam_callables`, `follow`) at
`h3_config.SUBJECT_TRACK`, on core's SAM 3.1 as core runs it, on the same frames as fed and nudged, each arm twice.

    <python> bench/subject_track_under_nudge.py run --clip C --second S --seconds T --width W --rate R --json J
    <python> bench/subject_track_under_nudge.py render --json J

THE READING, WRITTEN BEFORE THE FIRST RUN (2026-10-07):

- The floor is the two runs of one arm. They are expected to agree exactly (float16 was measured to repeat bit for bit);
  whatever they differ by is the floor, and a difference between arms counts only above it.
- A DECISION is, per shot: the cuts that bound it; whether the subject was taken there, as the pick, over the line, as
  the only person, or not at all; and which person, read as the overlap of the two arms' masks over the shot (under a
  half means another person or another place). Inside a shot: each time the track was seeded again, and each search that
  found nobody.
- The handling HOLDS under this change if every decision is the same in both arms. Likenesses may move; what matters is
  whether one crossed its line.
- It HOLDS BY A THIN MARGIN if the decisions are the same but some likeness moved by more than half of its distance to
  the line it was judged against. The table gives both numbers for every shot, so the thinnest is named.
- It DOES NOT HOLD for a shot whose decision differs. That is a finding about that line on that shot, not about the
  method: one clip, one stretch, one nudge.

What it cannot say: whether either arm's choice is the right person (no labels; the owner's typed corrections on this
stretch are not applied here, so this is the automatic pass alone), or anything about a render.

A shot's runner-up likeness is not kept by the node, so it is computed here with the node's own `sign`, `relative` and
`similarity` on the frame the shot's tile shows; the best computed that way must equal the node's recorded best, and a
row where it does not says so instead of a runner-up.

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
from sam3_parity_ladder import COMFY, REPO, boot, environment, load_core   # noqa: E402  (this folder's own tool)

#: inherited: `bench/sam3_precision_arms.py::NUDGE_SEED`, so the nudge is the independent session's, frame for frame.
NUDGE_SEED = 1234
ARMS = (("as fed", 0), ("as fed, again", 0), ("nudged", 1), ("nudged, again", 1))
#: reasoned: a second of frames after a regain is long enough to see whether the two arms hold the same place, and short
#: enough to end before the next loss on the hard stretch this was written for.
REGAIN_WINDOW = 24
#: inherited: `subject_track.PROBE_STRIDE`, the spacing of the frames the node looks at after a loss.
REGAIN_NEAR = 12


def nudged(u8: np.ndarray, levels: int) -> np.ndarray:
    """Every value moved by `levels` of 255 at random sign, seeded per frame (`bench/sam3_precision_arms.py::nudged`)."""
    if levels == 0:
        return u8
    out = np.empty_like(u8)
    for i in range(u8.shape[0]):
        sign = np.random.default_rng(NUDGE_SEED + i).integers(0, 2, size=u8.shape[1:], dtype=np.int8) * 2 - 1
        out[i] = np.clip(u8[i].astype(np.int16) + sign * levels, 0, 255).astype(np.uint8)
    return out


def cmd_run(a):
    torch = boot(float32=False)
    import server
    # the loader node the masked graphs use, so the frames are the lane's; it reads the server's instance at import
    server.PromptServer.instance = types.SimpleNamespace(prompt_queue=None, routes=None)
    sys.path.insert(0, str(COMFY / "custom_nodes" / "ComfyUI-VideoHelperSuite"))
    from videohelpersuite.load_video_nodes import LoadVideoFFmpegPath
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    import _h3pack.subject_track as st
    from h3_config import SEGMENTER, SUBJECT_TRACK

    frames = LoadVideoFFmpegPath().load_video(video=a.clip, force_rate=float(a.rate), custom_width=a.width, custom_height=0,
                                              frame_load_cap=int(round(a.seconds * a.rate)), start_time=float(a.second), format="AnimateDiff")[0][..., :3]
    u8 = (frames.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    del frames
    n, H, W = (int(v) for v in u8.shape[:3])
    core, clip = load_core(SEGMENTER)
    R = {"environment": environment(torch), "frames": {"clip": Path(a.clip).name, "second": a.second, "count": n, "rate": a.rate, "size": [W, H]},
         "settings": dict(SUBJECT_TRACK), "core_dtype": str(core.model.get_dtype()),
         "lines": {k: getattr(st, k) for k in ("MATCH_FLOOR", "PLAIN_FLOOR", "MATCH_THRESHOLD", "PLAIN_SAME", "REGAIN_SAME", "REGAIN_MARGIN", "MIN_GAP")},
         "arms": {}}
    packed = {}

    def one(levels: int):
        video = torch.from_numpy(nudged(u8, levels).astype(np.float32) / 255.0)
        steps = st.cut_scores(video, {})
        cut_at = st.auto_cuts(steps)
        cuts = st.find_cuts(steps, cut_at)
        detect, sign, track = st._sam_callables(core, clip, video, SUBJECT_TRACK["subject_phrase"], SUBJECT_TRACK["detection_threshold"],
                                                int(SUBJECT_TRACK["max_people"]), SUBJECT_TRACK["head_phrase"])
        with torch.no_grad():
            found = st.follow(n, cuts, SUBJECT_TRACK["pick"], None, None, detect, sign, track)
            mask = st.assemble(n, H, W, found.pieces) > 0.5
            # the likeness of every person on the frame each shot's tile shows, as `follow`'s `alike` computes it
            ranked = {}
            if found.pick_frame is not None:
                masks0, _ = detect(found.pick_frame)
                picked = next((s for s in found.shots if s.picked), None)
                mine = st._views(sign(found.pick_frame, masks0[picked.index])) if picked is not None and picked.index is not None else ()
                theirs = [st._views(sign(found.pick_frame, m)) for i, m in enumerate(masks0) if picked is None or i != picked.index]
                use = [k for k, v in enumerate(mine) if v is not None]
                centres = []
                for k in range(len(mine)):
                    have = [t[k] for t in theirs if t[k] is not None]
                    centres.append(torch.stack(have, dim=0).mean(dim=0) if have else None)
                reference = tuple(st.relative(v, c) for v, c in zip(mine, centres))
                for number, shot in enumerate(found.shots, 1):
                    if shot.picked or shot.index is None:
                        continue
                    sims = []
                    for m in detect(shot.shown)[0]:
                        views = st._views(sign(shot.shown, m))
                        if use and not any(views[k] is None for k in use):
                            sims.append(min(st.similarity(reference[k], st.relative(views[k], centres[k])) for k in use))
                    ranked[number] = sorted(sims, reverse=True)
        shots = []
        for number, s in enumerate(found.shots, 1):
            state = "the pick" if s.picked else "absent" if s.seed is None else "taken as the only person" if s.lone else "taken over the line"
            r = ranked.get(number, [])
            agrees = bool(r) and abs(r[0] - s.best) < 1e-6
            shots.append({"shot": number, "start": s.start, "end": s.end, "decision": state, "seed": s.seed, "shown": s.shown, "detections": s.candidates,
                          "best": round(float(s.best), 4), "runner_up": round(float(r[1]), 4) if agrees and len(r) > 1 else None,
                          "runner_up_note": "" if agrees or s.picked or s.index is None else "the recomputed best does not equal the node's",
                          "regained": [{"empty_from": x[0], "seeded_on": x[1], "detections": x[2], "best": round(x[3], 4), "next": round(x[4], 4)} for x in s.regained],
                          "searched_and_found_nobody": [list(x) for x in s.searched],
                          "probes_after_a_loss": len(s.probes),
                          # every frame looked at after a loss, taken or not: (frame, detections, best, the next person's)
                          "probes": [[int(x[0]), int(x[1]), round(float(x[2]), 4), round(float(x[3]), 4)] for x in s.probes],
                          "frames_with_a_mask": int(mask[s.start:s.end].flatten(1).any(1).sum())})
        return {"cut_line": round(float(cut_at), 4), "cuts": [int(c) for c in cuts], "pick_frame": found.pick_frame, "others_on_the_pick_frame": found.others,
                "match_line": round(float(found.match), 4), "shots": shots}, np.packbits(mask.numpy(), axis=-1)

    arms = [arm for arm in ARMS if not (a.no_repeats and arm[0].endswith("again"))]
    for name, levels in arms:
        R["arms"][name], packed[name] = one(levels)
        print(name, json.dumps({k: v for k, v in R["arms"][name].items() if k != "shots"}), flush=True)
        for s in R["arms"][name]["shots"]:
            print("   ", json.dumps(s), flush=True)

    def overlap(x, y, lo, hi):
        """Mean over the frames either arm has a mask on of intersection over union, and how many such frames."""
        vals = []
        for f in range(lo, hi):
            p, q = np.unpackbits(x[f], axis=-1), np.unpackbits(y[f], axis=-1)
            u = int((p | q).sum())
            if u:
                vals.append(int((p & q).sum()) / u)
        return (round(float(np.mean(vals)), 4) if vals else None), len(vals)

    R["between"] = {}
    for x, y in (("as fed", "as fed, again"), ("nudged", "nudged, again"), ("as fed", "nudged")):
        if x not in R["arms"] or y not in R["arms"]:
            continue
        ax, ay = R["arms"][x], R["arms"][y]
        row = {"same_cuts": ax["cuts"] == ay["cuts"], "same_pick_frame": ax["pick_frame"] == ay["pick_frame"],
               "match_line": [ax["match_line"], ay["match_line"]], "shots": []}
        if row["same_cuts"]:
            for sx, sy in zip(ax["shots"], ay["shots"]):
                iou, frames_any = overlap(packed[x], packed[y], sx["start"], sx["end"])
                line = ax["match_line"]
                row["shots"].append({"shot": sx["shot"], "decision": [sx["decision"], sy["decision"]], "same_decision": sx["decision"] == sy["decision"],
                                     "best": [sx["best"], sy["best"]], "best_moved": round(abs(sx["best"] - sy["best"]), 4),
                                     "distance_to_the_match_line": None if sx["decision"] == "the pick" else round(abs(sx["best"] - line), 4),
                                     "runner_up": [sx["runner_up"], sy["runner_up"]],
                                     "regained": [len(sx["regained"]), len(sy["regained"])],
                                     "mask_overlap": iou, "frames_either_has_a_mask": frames_any})
                # each time either arm seeded the track again: did the other do the same nearby, and did the two then
                # hold the same place? An overlap under a half after a regain is another person, or one arm with nobody.
                for arm, mine, theirs in ((x, sx, sy), (y, sy, sx)):
                    for g in mine["regained"]:
                        until = min(g["seeded_on"] + REGAIN_WINDOW, sx["end"])
                        after, frames_after = overlap(packed[x], packed[y], g["seeded_on"], until)
                        row.setdefault("regains", []).append({
                            "shot": sx["shot"], "arm": arm, "empty_from": g["empty_from"], "seeded_on": g["seeded_on"], "best": g["best"], "next": g["next"],
                            "lead": round(g["best"] - g["next"], 4),
                            "other_arm_seeded_within_a_stride": any(abs(o["seeded_on"] - g["seeded_on"]) <= REGAIN_NEAR for o in theirs["regained"]),
                            "overlap_of_the_two_arms_after_it": after, "frames_compared": frames_after})
        R["between"][f"{x} | {y}"] = row
    Path(a.json).write_text(json.dumps(R, indent=1))


def cmd_render(a):
    D = json.loads(Path(a.json).read_text())
    if a.key:
        D = D[a.key]
    f = D["frames"]
    print(f"`{f['clip']}` from {f['second']} s, {f['count']} frames at {f['rate']} a second, {f['size'][0]}x{f['size'][1]}; "
          f"the node's lines: {D['lines']}.\n")
    print("| arm | cuts found | pick frame | others on it | match line | shots taken | shots absent | times seeded again |\n|---|---|---|---|---|---|---|---|")
    for name, arm in D["arms"].items():
        taken = sum(s["decision"] != "absent" for s in arm["shots"])
        print(f"| {name} | {arm['cuts']} | {arm['pick_frame']} | {arm['others_on_the_pick_frame']} | {arm['match_line']} | {taken} | {len(arm['shots']) - taken} | "
              f"{sum(len(s['regained']) for s in arm['shots'])} |")
    if not D.get("between"):       # one arm only: its regains, with nothing to compare them with
        for name, arm in D["arms"].items():
            regains = [(s["shot"], g) for s in arm["shots"] for g in s["regained"]]
            if regains:
                print(f"\nEach time the `{name}` arm seeded the track again (the line is {D['lines']['REGAIN_SAME']}, the lead required {D['lines']['REGAIN_MARGIN']}):\n")
                print("| shot | empty from | seeded on | best | next person | lead |\n|---|---|---|---|---|---|")
                for shot, g in regains:
                    print(f"| {shot} | {g['empty_from']} | {g['seeded_on']} | {g['best']} | {g['next']} | {round(g['best'] - g['next'], 4)} |")
            for s in arm["shots"]:
                print(f"\nShot {s['shot']}: {s['frames_with_a_mask']} of {s['end'] - s['start']} frames with a mask; searched and found nobody over {s['searched_and_found_nobody']}.")
    for pair, row in D.get("between", {}).items():
        print(f"\n### {pair}\n\nSame cuts: {row['same_cuts']}; same pick frame: {row['same_pick_frame']}; match line {row['match_line'][0]} and {row['match_line'][1]}.\n")
        if not row["shots"]:
            print("The cuts differ, so the shots are not the same shots and are not compared one to one.")
            continue
        print("| shot | decision in each | same | best likeness in each | it moved by | its distance to the line | runner-up in each | seeded again in each | mask overlap | frames either has a mask |\n|---|---|---|---|---|---|---|---|---|---|")
        for s in row["shots"]:
            print(f"| {s['shot']} | {s['decision'][0]} / {s['decision'][1]} | {s['same_decision']} | {s['best'][0]} / {s['best'][1]} | {s['best_moved']} | "
                  f"{s['distance_to_the_match_line']} | {s['runner_up'][0]} / {s['runner_up'][1]} | {s['regained'][0]} / {s['regained'][1]} | {s['mask_overlap']} | {s['frames_either_has_a_mask']} |")
        for arm in (x_name for x_name in pair.split(" | ")):
            looked = [(sh["shot"], p) for sh in D["arms"][arm]["shots"] for p in sh.get("probes", [])]
            if looked:
                print(f"\nEvery frame the `{arm}` arm looked at after a loss (the line is {D['lines']['REGAIN_SAME']}, the lead required {D['lines']['REGAIN_MARGIN']}):\n")
                print("| shot | frame | detections | best | next person | lead | taken |\n|---|---|---|---|---|---|---|")
                for shot, (frame, found, best, nxt) in looked:
                    took = best >= D["lines"]["REGAIN_SAME"] and best - nxt >= D["lines"]["REGAIN_MARGIN"]
                    print(f"| {shot} | {frame} | {found} | {best} | {nxt} | {round(best - nxt, 4)} | {'yes' if took else 'no'} |")
        if row.get("regains"):
            print(f"\nEach time an arm seeded the track again (the line is {D['lines']['REGAIN_SAME']}, the lead required {D['lines']['REGAIN_MARGIN']}):\n")
            print("| shot | arm | empty from | seeded on | best | next person | lead | the other arm seeded within a stride | overlap of the two arms over the next second | frames compared |\n|---|---|---|---|---|---|---|---|---|---|")
            for g in sorted(row["regains"], key=lambda g: (g["shot"], g["seeded_on"], g["arm"])):
                print(f"| {g['shot']} | {g['arm']} | {g['empty_from']} | {g['seeded_on']} | {g['best']} | {g['next']} | {g['lead']} | {g['other_arm_seeded_within_a_stride']} | "
                      f"{g['overlap_of_the_two_arms_after_it']} | {g['frames_compared']} |")
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
    s.add_argument("--no-repeats", action="store_true", help="leave out the two repeat arms (repeats have been identical; they double the time)")
    s = sub.add_parser("render")
    s.set_defaults(fn=cmd_render)
    s.add_argument("--json", required=True)
    s.add_argument("--key", default="", help="when the json holds several runs, the one to print")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
