#!/usr/bin/env python3
"""Is a track that still has a plausible mask still on the SUBJECT? The likeness of each track to the subject's own seed.

`bench/results/2026-10-07_subject_alone_or_in_a_group.md` found that one subject seeded with the fifteen most prominent
people around them keeps a plausible mask several times longer than when followed alone. It could not say the mask is
still on the subject: a plausible mask on the person next to them counts, and neither the shape test nor the watch
(`subject_tracks.py`) can see that. This is the test that can: at each look the tracked mask is signed the way the
Subject Track signs a person (`subject_track.py::_sam_callables`'s `sign`: the image trunk's features under the mask's
top third, and under its head) and compared with the signature of the subject's own seed mask.

The group gives the test its own control for free. All sixteen tracks are compared with the SUBJECT's seed. The
subject's track should be the most like it; every other track is a deliberately wrong answer, and the nearest
neighbour's is the wrong answer a slipped track would most likely become. So, per look:

    own        the likeness of the subject's track to the subject's seed
    others     the likeness of every other track that has a plausible mask there
    first      whether own is the highest, and by how much it leads the best other

The node's likeness is the LOWER of two places compared, the top third of the mask and the head, and a person with no
head found scores as no match. The first run of this tool (2026-10-07) used only that and was blind: on most looks no
head was found for the subject's own track, so "no match" said nothing about who the track was on. So four measures
are reported, the same way each, and beside each how often it had nothing to compare:

    the top third alone          always there
    with a head, 16 on the frame the node's own measure (`max_people` heads asked on the whole frame)
    with a head, 64 on the frame the same with 64 heads asked
    with a head, from a crop     heads asked on a crop around the judged track's own box
    the lightness under the mask no model and no head: a coarse histogram of the lightness of the pixels under the mask,
                                 eroded a little, against the same under the subject's seed mask (histogram intersection,
                                 0 to 1). Always there. It is clothing, not identity: it tells a subject from neighbours
                                 only where they are lighter or darker than each other, and has no line of its own yet,
                                 so only "first among the tracks" is counted for it.

    <python> bench/subject_likeness_in_a_group.py run --clip C --second S --seconds T --width W --rate R --subject P --json J [--key K] [--shipped]
    <python> bench/subject_likeness_in_a_group.py render --json J

THE READING, WRITTEN BEFORE THE FIRST RUN (2026-10-07). The likeness test is usable as the tracker's "still the subject"
check only if (1) on the looks where the subject's track is plausible, own is over the Subject Track's regain line and
first among the tracks on most of them, in both the plain and the nudged run; and (2) THE CONTROL FAILS: the nearest
neighbour's track, judged as if it were the subject, is over the line and first on few or none. If the neighbour passes
about as often as the subject, likeness to the seed cannot tell them apart on this footage and the plausible frames of
the group result stay unverified. A look where own is NOT first is either a slipped track or a subject who turned away
from how the seed showed them; this tool cannot tell which, and says only how often it happens. No labels; one stretch.

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

SMALL = 252          # reasoned: the grid `subject_tracks.plausible` is asked on; it works in shares of the frame
SEED_FRAME = 4       # inherited: `subject_track.PROBE_OFFSET`
LOOK_EVERY = 6       # reasoned: twice as often as the node looks, to have enough looks on a six-second stretch
MOST = 16            # inherited: `h3_config.SUBJECT_TRACK["max_people"]`; one group of the tracker
SAME_OBJECT = 0.5    # inherited: core's line for "the same object" between detections (`comfy/ldm/sam3/tracker.py`)
HEAD_CROP_TIMES = 1.5  # reasoned: the crop a head is asked on is this many times the judged track's box, around its centre
MEASURES = ("the top third alone", "with a head, 16 on the frame", "with a head, 64 on the frame", "with a head, from a crop", "the lightness under the mask")
LIGHTNESS = MEASURES[4]
LIGHTNESS_BINS = 16    # reasoned: coarse enough that a pose change does not move it, fine enough to part light from dark
ERODE = 5              # reasoned: pixels; the mask's rim is where the neighbour and the background leak in


def cmd_run(a):
    torch = boot(float32=False)
    import torch.nn.functional as F

    import server
    server.PromptServer.instance = types.SimpleNamespace(prompt_queue=None, routes=None)
    sys.path.insert(0, str(COMFY / "custom_nodes" / "ComfyUI-VideoHelperSuite"))
    from videohelpersuite.load_video_nodes import LoadVideoFFmpegPath
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    from comfy.ldm.sam3.tracker import unpack_masks
    from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_VideoTrack

    import _h3pack.subject_track as st
    import sam31_corrections
    import subject_tracks as S
    from h3_config import SEGMENTER, SUBJECT_TRACK

    loaded = LoadVideoFFmpegPath().load_video(video=a.clip, force_rate=float(a.rate), custom_width=a.width, custom_height=0,
                                              frame_load_cap=int(round(a.seconds * a.rate)), start_time=float(a.second), format="AnimateDiff")[0][..., :3]
    u8 = (loaded.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    del loaded
    n, H, W = (int(v) for v in u8.shape[:3])
    stock, stock_clip = load_core(SEGMENTER)
    model, clip = (stock, stock_clip) if a.shipped else sam31_corrections.corrected(stock, stock_clip)
    line, lead = st.REGAIN_SAME, st.REGAIN_MARGIN
    R = {"environment": environment(torch), "frames": {"clip": Path(a.clip).name, "second": a.second, "count": n, "rate": a.rate, "size": [W, H]},
         "sam": "as ComfyUI ships it" if a.shipped else "corrected (sam31_corrections)", "device": str(torch.empty(0, device="cuda").device),
         "seed_frame": SEED_FRAME, "look_every": LOOK_EVERY,
         "line": line, "lead": lead, "one_run_per_arm": True, "runs": {}}

    # ONE subject and ONE set of seeds for both runs: found once on the frames as fed by SAM as it ships, as the group tool does
    plain = torch.from_numpy(u8.astype(np.float32) / 255.0)
    with torch.inference_mode():
        cond = stock_clip.encode_from_tokens_scheduled(stock_clip.tokenize(f"person:{MOST}"))
        out = SAM3_Detect.execute(stock, plain[SEED_FRAME:SEED_FRAME + 1], conditioning=cond, threshold=0.5, individual_masks=True)
    found = getattr(out, "args", out)[0].float().cpu()
    if found.shape[0] == 0:
        raise SystemExit("nothing detected on the seed frame")
    area = found.flatten(1).sum(1)
    subject = int(st.choose(found, [], a.subject))      # the node's own rule; `largest` is the shipped `pick`
    box = S.box_of(found[subject])
    R["subject"] = {"picked_by": a.subject, "box_on_the_seed_frame": None if box is None else [round(v, 3) for v in box]}
    flat = (found > 0.5).flatten(1).float()

    def overlap_of(i, j) -> float:
        inter = float((flat[i] * flat[j]).sum())
        return max(inter / max(float(flat[i].sum() + flat[j].sum()) - inter, 1.0), inter / max(min(float(flat[i].sum()), float(flat[j].sum())), 1.0))

    chosen = []
    for i in [subject] + [i for i in area.argsort(descending=True).tolist() if i != subject]:
        if not any(overlap_of(i, j) >= SAME_OBJECT for j in chosen):
            chosen.append(i)
        if len(chosen) == MOST:
            break
    seeds = found[chosen]
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    cx = (seeds * xs).flatten(1).sum(1) / seeds.flatten(1).sum(1).clamp(min=1)
    cy = (seeds * ys).flatten(1).sum(1) / seeds.flatten(1).sum(1).clamp(min=1)
    nearest = int(((cx - cx[0]) ** 2 + (cy - cy[0]) ** 2)[1:].argmin()) + 1 if seeds.shape[0] > 1 else None      # the row of the nearest neighbour
    R["seeded"] = int(seeds.shape[0])
    R["nearest_neighbour_row"] = nearest

    import comfy.model_management as mm
    import comfy.utils
    phrase, head, threshold = SUBJECT_TRACK["subject_phrase"], SUBJECT_TRACK["head_phrase"], SUBJECT_TRACK["detection_threshold"]

    for name, levels in (("as fed", 0), ("nudged", 1)):
        video = plain if levels == 0 else torch.from_numpy(nudged(u8, levels).astype(np.float32) / 255.0)
        _d, sign16, _t = st._sam_callables(model, clip, video, phrase, threshold, int(SUBJECT_TRACK["max_people"]), head)
        _d, sign64, _t = st._sam_callables(model, clip, video, phrase, threshold, 64, head)
        head_cond = clip.encode_from_tokens_scheduled(clip.tokenize(st.counted(head, int(SUBJECT_TRACK["max_people"]))))
        features = {}

        def trunk(f):
            if f not in features:
                mm.load_model_gpu(model)
                x = comfy.utils.common_upscale(video[f:f + 1, ..., :3].movedim(-1, 1), st.TRUNK_SIDE, st.TRUNK_SIDE, "bilinear", crop="disabled")
                out = model.model.diffusion_model.detector.backbone["vision_backbone"].trunk(x.to(device=mm.get_torch_device(), dtype=model.model.get_dtype()))
                features[f] = (out[-1] if isinstance(out, (list, tuple)) else out)[0].to(torch.float32).cpu()
            return features[f]

        def sign_crop(f, mask):
            """The node's two places, the head looked for on a crop around the mask's own box."""
            box = S.box_of(mask)
            top = st.signature(trunk(f), st.top_third(mask))
            if box is None:
                return top, None
            cx, cy, w, h = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H, (box[2] - box[0]) * W, (box[3] - box[1]) * H
            x0, x1 = int(max(cx - w * HEAD_CROP_TIMES / 2, 0)), int(min(cx + w * HEAD_CROP_TIMES / 2, W))
            y0, y1 = int(max(cy - h * HEAD_CROP_TIMES / 2, 0)), int(min(cy + h * HEAD_CROP_TIMES / 2, H))
            got = SAM3_Detect.execute(model, video[f:f + 1, y0:y1, x0:x1], conditioning=head_cond, threshold=float(threshold), individual_masks=True)
            part = getattr(got, "args", got)[0].to(torch.float32).cpu()
            heads = torch.zeros((part.shape[0], H, W), dtype=torch.float32)
            heads[:, y0:y1, x0:x1] = part
            found_head = st.head_of(mask, heads)
            return top, (None if found_head is None else st.signature(trunk(f), found_head))

        signers = {MEASURES[1]: sign16, MEASURES[2]: sign64, MEASURES[3]: sign_crop}

        def lightness(f, mask):
            """A histogram of the lightness of frame `f`'s pixels under `mask` ([H, W]), the mask eroded a little. None when it covers nothing."""
            on = mask > 0.5
            inner = (1 - F.max_pool2d(1 - on.float()[None, None], ERODE, stride=1, padding=ERODE // 2))[0, 0] > 0.5
            on = inner if bool(inner.any()) else on
            if not bool(on.any()):
                return None
            rgb = video[f][on]
            luma = rgb[:, 0] * 0.299 + rgb[:, 1] * 0.587 + rgb[:, 2] * 0.114
            hist = torch.histc(luma, bins=LIGHTNESS_BINS, min=0.0, max=1.0)
            return hist / hist.sum().clamp(min=1)

        seed_light = lightness(SEED_FRAME, seeds[0])
        with torch.inference_mode():
            # the subject's own seed, signed once per way of finding the head; the top third is the same in all
            seed_views = {m: st._views(fn(SEED_FRAME, seeds[0])) for m, fn in signers.items()}
            out = SAM3_VideoTrack.execute(video[SEED_FRAME:], model, initial_mask=seeds, conditioning=None, detection_threshold=0.5, max_objects=0, detect_interval=1)
            packed = getattr(out, "args", out)[0]["packed_masks"]
            looks = []
            for f in range(SEED_FRAME + LOOK_EVERY, n, LOOK_EVERY):
                t = f - SEED_FRAME
                rows = torch.stack([unpack_masks(packed[t:t + 1, k])[0] for k in range(seeds.shape[0])]).cpu()      # [K, 1008, 1008] bool
                ok = S.plausible(F.adaptive_max_pool2d(rows.float()[:, None], (SMALL, SMALL))[:, 0] > 0.5).tolist()
                full = (F.interpolate(rows.float()[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0] > 0.5).float()
                sims: dict = {m: [None] * len(ok) for m in MEASURES}       # None: nothing to compare (not plausible, or no head found)
                for k in range(len(ok)):
                    if not ok[k]:
                        continue
                    for m, fn in signers.items():
                        g, v = seed_views[m], st._views(fn(f, full[k]))
                        top = st.similarity(g[0], v[0])
                        sims[MEASURES[0]][k] = round(top, 4)
                        if g[-1] is not None and v[-1] is not None:
                            sims[m][k] = round(min(top, st.similarity(g[-1], v[-1])), 4)
                    light = lightness(f, full[k])
                    if light is not None and seed_light is not None:
                        sims[LIGHTNESS][k] = round(float(torch.minimum(light, seed_light).sum()), 4)

                def judged(row: int) -> dict:
                    """The track on `row` judged as if it were the subject, by each measure: its likeness to the SUBJECT's seed against every other plausible track's."""
                    out = {"plausible": bool(ok[row])}
                    for m in MEASURES:
                        own = sims[m][row]
                        others = [x for k, x in enumerate(sims[m]) if k != row and x is not None]
                        best_other = max(others) if others else None
                        has_line = m != LIGHTNESS        # the lightness has no line or margin of its own: only "first" means anything for it
                        out[m] = {"likeness": own, "best_other": best_other,
                                  "over_the_line": None if own is None else bool(own >= line) if has_line else True,
                                  "first": None if own is None else bool(best_other is None or own > best_other),
                                  "leads_by_the_margin": None if own is None else bool(best_other is None or own - best_other >= (lead if has_line else 0.0))}
                    return out
                looks.append({"frame": f, "plausible_tracks": int(sum(ok)), "subject": judged(0), **({"nearest_neighbour": judged(nearest)} if nearest is not None else {})})
                print(name, json.dumps({"frame": f, "plausible_tracks": int(sum(ok)), **{who: {m: looks[-1][who][m]["likeness"] for m in MEASURES} for who in ("subject", "nearest_neighbour") if who in looks[-1]}}), flush=True)

        def tally(who: str) -> dict:
            out = {}
            for m in MEASURES:
                mine = [x[who] for x in looks if who in x and x[who]["plausible"]]
                seen = [x[m] for x in mine if x[m]["likeness"] is not None]
                out[m] = {"looks_with_a_plausible_mask": len(mine), "nothing_to_compare": len(mine) - len(seen), "over_the_line": sum(x["over_the_line"] for x in seen),
                          "first": sum(x["first"] for x in seen), "over_the_line_and_first": sum(x["over_the_line"] and x["first"] for x in seen),
                          "over_the_line_and_leading_by_the_margin": sum(x["over_the_line"] and x["leads_by_the_margin"] for x in seen)}
            return out
        R["runs"][name] = {"the_seed_has_a_head": {m: seed_views[m][-1] is not None for m in signers}, "looks": looks,
                           "subject": tally("subject"), "nearest_neighbour_judged_as_the_subject": tally("nearest_neighbour")}
        print(name, json.dumps({k: v for k, v in R["runs"][name].items() if k != "looks"}), flush=True)
        f_ = Path(a.json)
        data = json.loads(f_.read_text()) if f_.exists() else {}
        data[a.key or f"{R['frames']['clip']} from {a.second} s"] = R
        f_.write_text(json.dumps(data, indent=1))
        torch.cuda.empty_cache()


def cmd_render(a):
    for name, D in json.loads(Path(a.json).read_text()).items():
        f = D["frames"]
        print(f"### {name}\n\n`{f['clip']}` from {f['second']} s, {f['count']} frames at {f['rate']} a second, {f['size'][0]}x{f['size'][1]}; SAM 3.1 {D['sam']}, on {D.get('device', 'the card')}; "
              f"{D['seeded']} seeded on frame {D['seed_frame']}, the subject first; a look every {D['look_every']} frames; the line {D['line']}, the margin {D['lead']}. One run per arm.\n")
        print("| run | measure | who is judged against the subject's seed | looks with a plausible mask | nothing to compare | over the line | first among the tracks | over the line and first | and leading by the margin |\n|---|---|---|---|---|---|---|---|---|")
        for run, r in D["runs"].items():
            for m in MEASURES:
                for who, label in (("subject", "the subject's own track"), ("nearest_neighbour_judged_as_the_subject", "THE CONTROL: the nearest neighbour's track")):
                    t = r[who][m]
                    lined = m != LIGHTNESS
                    print(f"| {run} | {m} | {label} | {t['looks_with_a_plausible_mask']} | {t['nothing_to_compare']} | {t['over_the_line'] if lined else 'no line'} | {t['first']} | "
                          f"{t['over_the_line_and_first'] if lined else 'no line'} | {t['over_the_line_and_leading_by_the_margin'] if lined else 'no line'} |")
        print()
        show = lambda x: "none" if x is None else f"{x:.3f}"       # noqa: E731
        for run, r in D["runs"].items():
            print(f"{run}, the subject's own track look by look (`none`: no head found to compare, or no plausible mask):\n")
            print("| frame | " + " | ".join(str(x["frame"]) for x in r["looks"]) + " |\n|---|" + "---|" * len(r["looks"]))
            print("| plausible | " + " | ".join("yes" if x["subject"]["plausible"] else "no" for x in r["looks"]) + " |")
            for m in MEASURES:
                print(f"| {m}: own | " + " | ".join(show(x["subject"][m]["likeness"]) for x in r["looks"]) + " |")
                print(f"| {m}: the best other track | " + " | ".join(show(x["subject"][m]["best_other"]) for x in r["looks"]) + " |")
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
    s.add_argument("--key", default="")
    s.add_argument("--subject", required=True, choices=["largest", "most central"], help="the node's `pick` rule that chooses the subject on the seed frame. Always named")
    s.add_argument("--shipped", action="store_true", help="SAM 3.1 as ComfyUI ships it; the default is through sam31_corrections, which the group result was strongest on")
    s = sub.add_parser("render")
    s.set_defaults(fn=cmd_render)
    s.add_argument("--json", required=True)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
