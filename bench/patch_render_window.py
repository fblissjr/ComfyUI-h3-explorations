#!/usr/bin/env python3
"""Redo one stretch of a finished masked render: build the one-window graph that patches it, and join the patch back.

    <python> bench/patch_render_window.py build --render R.mp4 --graph R.json --source SRC24.mov --source-first 100 \\
        --track-mask TRACK.mkv --parts-mask PARTS.mkv --window 290 294 --hole 320 540 --prompt-file P.txt --out DIR
    <python> bench/patch_render_window.py join --render R.mp4 --patch PATCH.mp4 --window 290 --hole 320 540 \\
        --parts-mask PARTS.mkv --out JOINED.mp4

**What it buys.** A long masked render that is right except for one stretch does not have to be rendered again, and
the stretch does not have to be continued from the window before it. The masked lane is run on the finished render
ITSELF as its source, for one window, with the hole open only on the frames to redo. Every other frame of the window
is the finished render's own picture, kept whole: the model is shown it as it is shown any kept plate, and the result
puts those pixels back. So the redraw is held at both ends by frames of what is already there, and there is no seam
to carry across. First used on 2026-10-08 (the masking board, finding `mhi-09`): one window, one seed.

**build** writes, into `--out`:
- the two mask clips for the window, lossless, closed on every frame outside the hole. They are cut from the mask
  videos of the finished render's own load (nothing is tracked again), with stray specks dropped by the tracker's
  own rule (`subject_tracks.drop_specks`) and the part cut to the cleaned subject as the part node cuts it
  (`video_mask.select_part`), so a render made before those rules existed is patched with them;
- the motion video, when the finished render's graph used one: the SOURCE's frames for the window, cut by the pack's
  own `video_mask.motion_reference` with the settings of that graph, on every frame of the window;
- `patch.json`, the finished render's graph changed in the fewest places that make it this window, and the list of
  those places, printed.
It also says which frames the hole really covers once the model's frame grouping has had its say, and refuses a
window length the Song node would not render as one window.

**join** writes the finished render with the hole's frames taken from the patch and nothing else, by frame index,
with the render's own audio copied. It does not join anything itself: it hands `bench/assemble_delivery.py` a table
of one row (the hole's frames from the patch, over the render as the source), so there is one joiner, the file is
checked by decode as every delivery is (`<out>.check.json`), and a render written before the song node tagged its
files is brought to the patch's colour. A plain concatenation of such a render and a patch written after left the
hole's frames two to three levels off in every channel, in a file with no tag (measured 2026-10-10;
`bench/check_patch_render_window.py` holds the case). Then it prints, for the frames either side of each end of the hole, the patch against
the render inside the region and on the plate, the frame-to-frame change inside the region for both, and the mean
brightness inside the mask for both. On a shut frame the first number is one more encode and nothing else: that is
the proof the ends came back as the render's own pixels. A step in the brightness at the hole's first and last frames
is the redraw's own tone against the render it replaces, and is what a blend in time would hide.

**Three rules it was built on**, each from that first use:
- *Open the hole where the render and the source are in the same pose.* The first redrawn frame then agrees with the
  frame pinned before it and with the motion video. Nothing here can check it: it is a look at the two.
- *Put the hole's ends on the edges of the model's frame groups.* A window's frames are grouped 1, 4, 4, 4, 4 and
  again, and a group regenerates whole when any of its frames is in the hole (`video_mask.token_mask`). `build`
  prints the frames really covered and says when they are more than was asked.
- *The motion video is appended as a reference, not asked of the Masked Source.* That node cuts its motion video from
  its own source, and here the source is the finished render, whose movement is what is being redone. So the node's
  own `motion_reference` is turned off and the clip made here is appended as the reference video.

**What it does not do.** It does not queue anything, look at a clip, or choose the window, the hole or the text. The
part mask it takes is the finished render's: where that was wrong the patch is wrong the same way. A render whose
Masked Source finds its part with a SAM phrase (`head and hair`) is refused: that needs the segmenter again.
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, bootstrap  # noqa: E402

SONG, SOURCE = "MiniMaxH3AudioFreezeSong", "MiniMaxH3MaskedSource"
LOADER, TO_MASK, APPEND_VIDEO = "VHS_LoadVideoFFmpegPath", "ImageToMask", "MiniMaxH3AppendRefVideo"
#: How far before a frame, in frames, a loader is asked to start so that the frame is the first it yields. Reasoned:
#: a time between two frames cannot land one off, where a time on the frame depends on rounding inside ffmpeg.
#: Measured once, 2026-10-08: the first frames the loader's own seek yields from such a time were the asked ones.
SEEK_BACK = 0.4
#: The patch window's own encode. Reasoned: its kept frames are the render's pixels encoded once more, so the loss
#: is kept low; the join encodes again at `JOIN_CRF`.
PATCH_CRF = 12
JOIN_CRF = 12
#: Frames either side of each end of the hole that `join` reports. Reasoned: two frame groups each way.
REPORT_REACH = 8
#: A lossless grey clip for a mask and a lossless colour clip for the motion video. Inherited from the lane's own
#: saved masks (ffv1), so nothing is lost between the saved mask and the node.
FFV1 = ["-c:v", "ffv1", "-level", "3"]


def _pack(name: str):
    """A root module of the pack, loaded as the checks load it (a stand-in package, so its relative imports work)."""
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    return importlib.import_module(f"_h3pack.{name}")


def probe(path: str) -> tuple[int, int, float, int]:
    """(width, height, frames a second, frames) of a video's first video stream."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den), int(s["nb_read_packets"])


def start_time(frame: int, rate: float) -> float:
    return round(max(frame - SEEK_BACK, 0.0) / rate, 6)


def read(path: str, first: int, count: int, rate: float, size: tuple[int, int], pix: str = "gray", vf: str = "null"):
    """`count` frames of a video from frame `first`, as a uint8 array [count, h, w(, 3)]."""
    import numpy as np
    w, h = size
    pre = ["-ss", f"{start_time(first, rate):.6f}"] if first else []
    raw = subprocess.run(["ffmpeg", "-v", "error", *pre, "-i", path, "-vf", vf, "-frames:v", str(count), "-f", "rawvideo",
                          "-pix_fmt", pix, "-"], capture_output=True, check=True).stdout
    shape = (-1, h, w) if pix == "gray" else (-1, h, w, 3)
    got = np.frombuffer(raw, np.uint8).reshape(shape)
    if got.shape[0] != count:
        raise SystemExit(f"{path}: {got.shape[0]} frame(s) from frame {first} where {count} were asked for")
    return got


def write(path: str, frames, rate: float, pix_in: str, pix_out: str) -> None:
    h, w = frames.shape[1], frames.shape[2]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{w}x{h}", "-r", f"{rate:g}",
                    "-i", "-", *FFV1, "-pix_fmt", pix_out, path], input=frames.tobytes(), check=True)


def load_graph(path: str) -> dict:
    """An API graph from a json file, or from the `prompt` a render's metadata PNG carries."""
    if path.lower().endswith(".png"):
        from PIL import Image
        return json.loads(Image.open(path).info["prompt"])
    graph = json.load(open(path, encoding="utf-8"))
    return graph.get("prompt", graph)


def the_one(graph: dict, class_type: str) -> str:
    ids = [nid for nid, node in graph.items() if node.get("class_type") == class_type]
    if len(ids) != 1:
        raise SystemExit(f"the graph holds {len(ids)} {class_type} node(s); this tool patches a graph with exactly one")
    return ids[0]


def reachable(graph: dict, roots: list[str]) -> set[str]:
    seen, todo = set(), list(roots)
    while todo:
        nid = todo.pop()
        if nid in seen or nid not in graph:
            continue
        seen.add(nid)
        for value in graph[nid]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                todo.append(value[0])
    return seen


def show(value) -> str:
    if isinstance(value, str) and len(value) > 90:
        return f"<text, {len(value)} characters>"
    return json.dumps(value)


def print_diff(old: dict, new: dict) -> None:
    print("changes against the finished render's graph (node[id].input: render -> patch):")
    for nid in sorted(set(old) | set(new), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x)):
        if nid not in new:
            print(f"   removed  {old[nid]['class_type']}[{nid}]")
        elif nid not in old:
            ins = {k: (Path(v).name if k == "video" else v) for k, v in new[nid]["inputs"].items()}
            print(f"   added    {new[nid]['class_type']}[{nid}] {show(ins)}")
        elif old[nid]["class_type"] != new[nid]["class_type"]:
            print(f"   replaced {old[nid]['class_type']}[{nid}] by {new[nid]['class_type']} "
                  f"{show({k: (Path(v).name if k == 'video' else v) for k, v in new[nid]['inputs'].items()})}")
        else:
            a, b = old[nid]["inputs"], new[nid]["inputs"]
            for key in sorted(set(a) | set(b)):
                if a.get(key, "<absent>") != b.get(key, "<absent>"):
                    one, two = a.get(key, "<absent>"), b.get(key, "<absent>")
                    if key == "video":
                        one, two = Path(str(one)).name, Path(str(two)).name
                    print(f"   {new[nid]['class_type']}[{nid}].{key}: {show(one)} -> {show(two)}")


def build(args) -> int:
    import numpy as np
    bootstrap(cpu=True)
    import torch
    vm, tracks, plan = _pack("video_mask"), _pack("subject_tracks"), _pack("loop_plan")
    first, length = args.window
    hole_a, hole_b = args.hole
    if not first <= hole_a <= hole_b < first + length:
        raise SystemExit(f"the hole {hole_a}-{hole_b} is not inside the window {first}-{first + length - 1}")
    w, h, rate, total = probe(args.render)
    if first + length > total:
        raise SystemExit(f"the window ends at frame {first + length - 1} and the render has {total} frames")
    old = load_graph(args.graph)
    graph = json.loads(json.dumps(old))
    song_id = the_one(graph, SONG)
    sinks = [song_id] + [nid for nid, node in graph.items()
                         if any(isinstance(v, list) and len(v) == 2 and v[0] == song_id for v in node["inputs"].values())]
    used_before = reachable(graph, sinks)
    song = graph[song_id]["inputs"]
    source_id = song["source"][0]
    if graph[source_id]["class_type"] != SOURCE:
        raise SystemExit(f"the Song node's `source` is a {graph[source_id]['class_type']}, not a {SOURCE}")
    src = graph[source_id]["inputs"]
    replace, motion = src.get("replace", vm.REPLACE_WHOLE), src.get("motion_reference", vm.MOTION_NONE)
    if replace == vm.REPLACE_PART:
        raise SystemExit(f"the render's Masked Source finds its part with a phrase (`{replace}`): that needs the segmenter, "
                         "and this tool tracks and detects nothing")
    if replace == vm.REPLACE_PARTS and not args.parts_mask:
        raise SystemExit(f"the render's Masked Source replaces `{replace}`: give --parts-mask, the part mask of its load")
    # one window: the planner has the last word on whether this length renders as one
    planned = [n for _i, lengths in plan.plan_windows(length, int(song["window_frames"]), int(song["context_frames"]), ())
               for n in lengths]
    if planned != [length]:
        raise SystemExit(f"{length} frames would render as windows {planned}, not one: use one of {plan.CHAIN_LENGTHS} "
                         f"up to window_frames {song['window_frames']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mask_rate = probe(args.track_mask)[2]
    track = torch.from_numpy((read(args.track_mask, first, length, mask_rate, (w, h)) > 127).astype(np.float32))
    track, speck_frames, speck_px = tracks.drop_specks(track, in_place=True)
    if speck_frames:
        print(f"stray specks dropped from the track mask: {speck_px} px on window frame(s) {speck_frames}")
    settled = track
    if replace == vm.REPLACE_PARTS:
        parts = torch.from_numpy((read(args.parts_mask, first, length, probe(args.parts_mask)[2], (w, h)) > 127).astype(np.float32))
        settled = vm.select_part(track, parts, int(src.get("part_margin", vm.PART_MARGIN)))
        off = (parts - settled).flatten(1).sum(dim=1)
        if float(off.sum()):
            print(f"part pixels off the cleaned subject dropped: {int(off.sum())} px on window frame(s) "
                  f"{[int(i) for i in off.nonzero().flatten()]}")
    shut = torch.ones(length, dtype=torch.bool)
    shut[hole_a - first:hole_b - first + 1] = False
    track_hole, settled_hole = track.clone(), settled.clone()
    track_hole[shut] = 0.0
    settled_hole[shut] = 0.0
    as_bytes = lambda m: (m > 0.5).to(torch.uint8).mul(255).numpy()   # noqa: E731
    write(str(out / "patch_track_mask.mkv"), as_bytes(track_hole), rate, "gray", "gray")
    if replace == vm.REPLACE_PARTS:
        write(str(out / "patch_parts_mask.mkv"), as_bytes(settled_hole), rate, "gray", "gray")

    # which frames the hole really covers: the region as the Song node will make it
    record = {"mask": settled_hole, "grow_pixels": int(src.get("grow_pixels", vm.GROW_PIXELS)),
              "feather_pixels": int(src.get("feather_pixels", 0)), "grow_by": src.get("grow_by", vm.GROW_FIXED),
              "subject_area": vm.area_share(settled_hole)}
    lat_t = next(t for t in range(1, 4 * length) if sum(vm.run_lengths(t)) == length)
    tokens = vm.token_mask(vm.grow(settled_hole, vm.source_margins(record, 0, length, w * h)), lat_t, h // 16, w // 16)
    covered, at = [], 0
    for step, run in enumerate(vm.run_lengths(lat_t)):
        if float(tokens[step].sum()):
            covered += range(at, at + run)
        at += run
    if not covered:
        raise SystemExit("the mask is empty on every frame of the hole: nothing would regenerate")
    real_a, real_b = first + covered[0], first + covered[-1]
    print(f"the hole asked for is frames {hole_a}-{hole_b}; it really covers {real_a}-{real_b} ({len(covered)} frames); "
          f"{100.0 * float(tokens.mean()):.1f}% of the window's video tokens regenerate")
    if (real_a, real_b) != (hole_a, hole_b):
        print(f"   WIDER than asked by {hole_a - real_a} frame(s) before and {real_b - hole_b} after: an end is inside a frame "
              "group. Move it to a group's edge, or take these frames from the patch when joining")

    # the graph: the render as the source, the mask clips for the tracker and the part node, one window
    old_loader_id = src["frames"][0]
    fmt = graph[old_loader_id]["inputs"].get("format", "AnimateDiff")

    def loader(path: str, start: float) -> dict:
        return {"class_type": LOADER,
                "inputs": {"video": str(Path(path).resolve()), "force_rate": 0, "custom_width": w, "custom_height": h,
                           "frame_load_cap": length, "start_time": start, "format": fmt}}

    fresh = iter(range(max(int(n) for n in graph if n.isdigit()) + 1, 10 ** 9))
    graph[old_loader_id] = loader(args.render, start_time(first, rate))
    track_id, track_mask_id = str(next(fresh)), str(next(fresh))
    graph[track_id] = loader(str(out / "patch_track_mask.mkv"), 0.0)
    graph[track_mask_id] = {"class_type": TO_MASK, "inputs": {"image": [track_id, 0], "channel": "red"}}
    src["mask"] = [track_mask_id, 0]
    if replace == vm.REPLACE_PARTS:
        parts_id, parts_mask_id = str(next(fresh)), str(next(fresh))
        graph[parts_id] = loader(str(out / "patch_parts_mask.mkv"), 0.0)
        graph[parts_mask_id] = {"class_type": TO_MASK, "inputs": {"image": [parts_id, 0], "channel": "red"}}
        src["parts"] = [parts_mask_id, 0]
    for key in ("segmenter", "segmenter_clip", "shot_table"):
        src.pop(key, None)
    src["motion_reference"] = vm.MOTION_NONE

    if motion != vm.MOTION_NONE:
        if motion == vm.MOTION_ZOOM:
            raise SystemExit("the render's motion video is zoomed in, which needs the subject's box per shot: not built here")
        if not args.source:
            raise SystemExit(f"the render's graph has a motion video (`{motion}`): give --source and --source-first")
        source_rate = probe(args.source)[2]
        ar = w / h
        crop = f"crop=if(gt({ar}\\,a)\\,iw\\,ih*{ar}):if(gt({ar}\\,a)\\,iw/{ar}\\,ih),scale={w}:{h}"   # the loader's own
        pixels = torch.from_numpy(read(args.source, args.source_first + first, length, source_rate, (w, h), "rgb24", crop).copy())
        clip = vm.motion_reference(pixels.to(torch.float32) / 255.0, settled, motion, int(src.get("motion_short_edge", h)),
                                   int(src.get("grow_pixels", vm.GROW_PIXELS)) // 2)
        write(str(out / "patch_motion_video.mkv"), (clip.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy(), rate,
              "rgb24", "bgr0")
        motion_id, append_id = str(next(fresh)), str(next(fresh))
        graph[motion_id] = loader(str(out / "patch_motion_video.mkv"), 0.0)
        graph[motion_id]["inputs"].update({"custom_width": int(clip.shape[2]), "custom_height": int(clip.shape[1])})
        appended = {"frames": [motion_id, 0], "video_info": [motion_id, 3], "use_vae": bool(src.get("motion_vae", False))}
        if song.get("references") is not None:
            appended["references"] = song["references"]
        graph[append_id] = {"class_type": APPEND_VIDEO, "inputs": appended}
        song["references"] = [append_id, 0]
        print(f"motion video: `{motion}` cut from the source's frames {args.source_first + first}-"
              f"{args.source_first + first + length - 1}, {int(clip.shape[2])}x{int(clip.shape[1])}, on every frame of the window")

    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip() if args.prompt_file else ""
    song.update({"extent": "first_seconds", "extent.seconds": length / rate, "timeline": "", "crf": PATCH_CRF,
                 "filename_prefix": args.prefix, "prompt": prompt or "PROMPT NOT FILLED: give --prompt-file"})
    song.pop("continue_from", None)
    # only what this rewiring left with nothing to feed goes: a node the render's graph already left unwired stays
    keep = reachable(graph, sinks)
    for nid in [n for n in graph if n in used_before and n not in keep]:
        del graph[nid]
    (out / "patch.json").write_text(json.dumps(graph, indent=1), encoding="utf-8")
    print_diff(old, graph)
    print(f"wrote {out / 'patch.json'}: frames {first}-{first + length - 1} of the render as one window of {length}, "
          f"loaded from {start_time(first, rate)} s; the prompt is {'filled' if prompt else 'NOT FILLED, do not queue it'}")
    return 0


def _widen(mask, pixels: int):
    """A [H, W] bool mask widened by `pixels` each way, as a square: two one-way pools, so the cost follows the width."""
    import torch.nn.functional as F
    m = mask[None, None].float()
    m = F.max_pool2d(m, (1, 2 * pixels + 1), 1, (0, pixels))
    return F.max_pool2d(m, (2 * pixels + 1, 1), 1, (pixels, 0))[0, 0] > 0.5


def join(args) -> int:
    import torch
    first = args.window[0]
    hole_a, hole_b = args.hole
    w, h, rate, total = probe(args.render)
    pw, ph, patch_rate, patch_total = probe(args.patch)
    if (pw, ph) != (w, h) or not first <= hole_a <= hole_b < first + patch_total:
        raise SystemExit(f"the patch is {pw}x{ph}, {patch_total} frames from {first}; the render is {w}x{h} and the hole {hole_a}-{hole_b}")
    table = Path(args.out + ".table.txt")
    table.write_text(f"{hole_a}-{hole_b} {args.patch} {first}\n")
    try:
        done = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "assemble_delivery.py"), "--source", args.render,
                               "--table", str(table), "--span", f"0-{total - 1}", "--crf", str(args.crf), "--out", args.out,
                               "--note", f"patch_render_window join: render frames {hole_a}-{hole_b} from a patch that starts on frame {first}"],
                              capture_output=True, text=True)
    finally:
        table.unlink(missing_ok=True)
    if done.returncode != 0:
        raise SystemExit(f"bench/assemble_delivery.py did not pass the joined file (its record: {args.out}.check.json):\n"
                         + (done.stdout[-1200:] or done.stderr[-1200:]))
    joined = probe(args.out)[3]
    print(f"wrote {args.out}: {joined} frames (the render has {total}); frames {hole_a}-{hole_b} from the patch, the rest the render's; "
          f"checked by decode, record at {args.out}.check.json")

    luma = torch.tensor([0.299, 0.587, 0.114])
    grow = int(args.grow)
    for name, edge in (("the hole's first frame", hole_a), ("the first frame after the hole", hole_b + 1)):
        lo, hi = max(edge - REPORT_REACH, first), min(edge + REPORT_REACH, first + patch_total - 1)
        n = hi - lo + 1
        patch = torch.from_numpy(read(args.patch, lo - first, n, patch_rate, (w, h), "rgb24").copy()).float()
        render = torch.from_numpy(read(args.render, lo, n, rate, (w, h), "rgb24").copy()).float()
        mask = None
        if args.parts_mask:
            mask = torch.from_numpy(read(args.parts_mask, lo, n, probe(args.parts_mask)[2], (w, h)) > 127)
        print(f"== {name}: {edge}; frames {lo}-{hi}")
        print(" frame  hole | patch against render: in region / on plate | change from the frame before, in region: patch  render"
              " | brightness inside the mask: patch  render  (patch - render)")
        for i in range(n):
            f = lo + i
            if mask is not None:
                region = _widen(mask[i], grow)
                plate = ~_widen(region, 32)
                inside = mask[i]
            else:
                region = plate = inside = torch.ones((h, w), dtype=torch.bool)
            diff = (patch[i] - render[i]).abs().mean(dim=-1)
            line = f" {f:5d}  {'OPEN' if hole_a <= f <= hole_b else 'shut'} | {float(diff[region].mean()):6.2f} / {float(diff[plate].mean()):5.2f} |"
            if i:
                both = region
                if mask is not None:
                    both = region | _widen(mask[i - 1], grow)
                step = lambda x: float(((x[i] @ luma) - (x[i - 1] @ luma)).abs()[both].mean())   # noqa: E731
                line += f" {step(patch):6.2f} {step(render):6.2f} |"
            else:
                line += "      -      - |"
            if bool(inside.any()):
                lp, lr = float((patch[i] @ luma)[inside].mean()), float((render[i] @ luma)[inside].mean())
                line += f" {lp:7.2f} {lr:7.2f} ({lp - lr:+.2f})"
            print(line)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build", help="write the mask clips, the motion video and the one-window graph")
    b.add_argument("--render", required=True, help="the finished render's video")
    b.add_argument("--graph", required=True, help="its API graph, as json or as the metadata PNG saved beside it")
    b.add_argument("--source", help="the source video the render was made from, at the render's frame rate")
    b.add_argument("--source-first", type=int, default=0, help="the source frame that is the render's frame 0")
    b.add_argument("--track-mask", required=True, help="the tracked mask of the render's load, a grey video, one frame a frame")
    b.add_argument("--parts-mask", help="the part mask of the same load, when the render replaced the wired parts")
    b.add_argument("--window", required=True, nargs=2, type=int, metavar=("FIRST", "LENGTH"), help="in the render's frames")
    b.add_argument("--hole", required=True, nargs=2, type=int, metavar=("FIRST", "LAST"), help="in the render's frames, inclusive")
    b.add_argument("--prompt-file", help="the text for this window; without it the graph says so and must not be queued")
    b.add_argument("--prefix", default="Video/patch/patch", help="the Song node's filename_prefix for the patch")
    b.add_argument("--out", required=True, help="a folder for the clips and patch.json")
    j = sub.add_parser("join", help="put the hole's frames back into the render and report both ends")
    j.add_argument("--render", required=True)
    j.add_argument("--patch", required=True, help="the patch window's video")
    j.add_argument("--window", required=True, nargs=1, type=int, metavar="FIRST", help="the render frame the patch starts on")
    j.add_argument("--hole", required=True, nargs=2, type=int, metavar=("FIRST", "LAST"))
    j.add_argument("--parts-mask", help="the mask of the render's load, for the region the report reads; without it, whole frames")
    j.add_argument("--grow", type=int, default=64, help="the margin the region was grown by, for the report")
    j.add_argument("--crf", type=int, default=JOIN_CRF)
    j.add_argument("--out", required=True, help="the joined video")
    args = p.parse_args()
    return build(args) if args.mode == "build" else join(args)


if __name__ == "__main__":
    sys.exit(main())
