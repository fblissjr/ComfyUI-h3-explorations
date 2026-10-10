#!/usr/bin/env python3
"""`bench/recomposite_window.py` on a window made for the purpose: the frames it says a change moves are the frames it moves.

The tool's answer is trusted in place of a render, so each case is a way it could say the wrong frames: reading
the source from the wrong frame of the clip, laying with other settings than the ones asked, counting the trimmed
context as written, rebuilding a region that is not the render's, or decoding again where a decode was kept. The
window is small (a canvas of a few latent cells, seven latent steps) and its "decode" is handed in, so no model
runs: the source clip is lossless, the subject is on the frames before a cut that falls inside a latent step, and
the stand-in decode is the source a few levels off with a bright block on every frame.

1. **A window with its saved region.** Laid as rendered against laid with `cuts=none`: the frames that differ are
   the frames of the split step across the cut, counted in the clip's own frame numbers through `--source-first`
   and the window's first frame; each is the source bit for bit under the gate; frames before the trim are not
   counted as written; and the first laying is within a level of the window's own video where that video holds
   it, and far from it when the trim is read one frame off (the control).
2. **The settings.** `--as-rendered cuts=none` turns the pair round and names the same frames; laid with the
   whole region in place of only what changed, the frames that differ are the ones with a region and no cut
   beside them, and the gated ones stay the source under both; an unknown key is refused.
3. **The decode is kept.** With none kept the decode is asked for once and saved; the next run reads it and asks
   for none; both runs give the same rows.
4. **A box, and the step a frame is in.** A box over the block the decode draws: on the frames across the cut the
   decode is far from the source there and the laid frame is the source, with none of the box kept; on a frame
   the subject is on the laid box is the decode's own. The rows of the split step say it is split, and no other.
5. **A region rebuilt from a capture.** For a render that saved none: the mask read back from a capture, the
   settings from its graph, the cuts from its shot table, the first frame from the stored `next_start` and the
   context. It names the frames the saved region names, with no cell differing; a capture whose region is one
   cell off says one cell a frame; and a run that wires `keep` is refused.

Needs ffmpeg on the path. No model, no card, no server.

    <comfy venv python> bench/check_recomposite_window.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import bootstrap, case, finish, needs  # noqa: E402

needs("ffmpeg and ffprobe on the path", bool(shutil.which("ffmpeg") and shutil.which("ffprobe")))
bootstrap(cpu=True)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import recomposite_window as rw  # noqa: E402

vm = rw._pack("video_mask")
WORK = Path(tempfile.mkdtemp(prefix="check_recomposite_window_"))
LATENT = (1, 24, 7, 3, 4)                           # seven latent steps on a canvas of three by four cells
W, H = LATENT[4] * rw.CELL, LATENT[3] * rw.CELL
RUNS = vm.run_lengths(LATENT[2])
FRAMES = sum(RUNS)
STARTS = [sum(RUNS[:k]) for k in range(len(RUNS))]
STEP = next(i for i, n in enumerate(RUNS) if i >= 2 and n >= 3)
CUT = STARTS[STEP] + 1                              # one frame into a step: the subject's shot ends here
ACROSS = list(range(CUT, STARTS[STEP] + RUNS[STEP]))
LEAD, FIRST, TRIM = 3, 0, 5                         # the load starts on the clip's frame 3; the window's video leaves off 5
MARGIN, FEATHER = 8, 2


def lossless(path: Path, frames: np.ndarray, codec: list[str]) -> None:
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24",
                            "-i", "-", *codec, str(path)], stdin=subprocess.PIPE)
    assert enc.stdin is not None
    enc.stdin.write(frames.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0, f"ffmpeg could not write {path}"


def make() -> dict:
    """The clip, the window's stored latent, its saved region and the stand-in decode."""
    rng = np.random.default_rng(20261010)
    clip = rng.integers(40, 200, size=(LEAD + FRAMES, H, W, 3), dtype=np.uint8)
    clip[LEAD + CUT:] //= 2                         # another shot after the cut
    lossless(WORK / "clip.mkv", clip, ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp"])
    pixels = torch.from_numpy(clip[LEAD:]).float() / 255.0
    mask = torch.zeros(FRAMES, H, W)
    mask[:CUT, 16:32, 16:40] = 1.0
    tokens = vm.token_mask(vm.grow(mask, MARGIN), *LATENT[2:])
    settings = {"composite": vm.COMPOSITE_CHANGED, "feather_pixels": FEATHER, "change_threshold": vm.CHANGE_THRESHOLD,
                "cuts": [FIRST + CUT], "grow_pixels": MARGIN, "grow_by": vm.GROW_FIXED, "replace": vm.REPLACE_WHOLE,
                "edge": vm.EDGE_TOKENS}
    decode = pixels + 4.0 / 255.0                   # a decode is never the source: a few levels off everywhere, under the
    decode[:, 18:30, 20:36] = 0.95                  # change threshold; and the new subject, on every frame, the far side too
    folder = WORK / "r_windows"
    folder.mkdir()
    window = folder / "r_window_1.safetensors"
    save_file({"stream_0": torch.zeros(LATENT), "stream_1": torch.zeros(1, 2, 2, 4)}, str(window),
              metadata={"key": "k", "trim": str(TRIM), "next_start": repr((FRAMES - TRIM) / rw.FPS), "nested": "True",
                        "written": str(FRAMES - TRIM)})
    vm.save_window_region(str(folder / "r_window_1_region.npz"), mask, tokens, settings, MARGIN, FIRST, TRIM)
    return {"window": window, "pixels": pixels, "mask": mask, "tokens": tokens, "settings": settings, "decode": decode}


MADE = make()


def args(out: str, **more) -> argparse.Namespace:
    base = dict(window=str(MADE["window"]), source=str(WORK / "clip.mkv"), source_first=LEAD, region=None, capture=None,
                run=None, shots=None, context=None, first_frame=None, as_rendered=None, set=None, vae="none", device="cpu",
                out=str(WORK / out), box=None, source_frames=None)
    return argparse.Namespace(**{**base, **more})


def handed(calls: list):
    def decode(path, _vae, _device):
        calls.append(path)
        return MADE["decode"].clone()
    return decode


def saved_region() -> str:
    calls = []
    got = rw.run(args("a", set=["cuts=none"]), decode=handed(calls))["summary"]
    want = [LEAD + FIRST + f for f in ACROSS]
    assert got["source_frames_differing"] == want, f"the frames that differ are {got['source_frames_differing']}, not {want}"
    assert got["frames_written"] == FRAMES - TRIM, got["frames_written"]
    assert got["source_frames_differing_and_written"] == [f for f in want if f - LEAD - FIRST >= TRIM], got
    rows = list(__import__("csv").DictReader(open(WORK / "a" / "r_window_1.csv")))
    assert all(int(r["as_rendered_is_source"]) == 1 and int(r["changed_is_source"]) == 0 for r in rows if int(r["frame"]) in ACROSS), \
        "a frame across the cut is not the source under the gate, or is the source without it"
    assert [int(r["written"]) for r in rows] == [0] * TRIM + [1] * (FRAMES - TRIM), "the trimmed frames are counted as written"
    assert got["as_rendered_off_file_levels"] is None, "a comparison with a video that is not there"
    # the window's own video: the first laying, from the trim on
    # the tool lays the decode it kept (half floats), so the file is made from the same halves
    kept = torch.from_numpy(np.load(WORK / "a" / "decoded_r_window_1.npy")).float()
    laid = vm.lay_window(kept, MADE["pixels"], MADE["tokens"], MADE["mask"], MADE["settings"], MARGIN, FIRST)[0]
    video = (laid[TRIM:].clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    lossless(MADE["window"].with_suffix(".mp4"), video, ["-c:v", "libx264rgb", "-qp", "0"])
    off = rw.run(args("a"), decode=handed(calls))["summary"]["as_rendered_off_file_levels"]
    assert off and off["frames"] == FRAMES - TRIM and off["most"] <= 0.5, f"the first laying is {off} levels from its own video"
    lossless(MADE["window"].with_suffix(".mp4"), video[1:], ["-c:v", "libx264rgb", "-qp", "0"])        # the control: one frame off
    slipped = rw.run(args("a"), decode=handed(calls))["summary"]["as_rendered_off_file_levels"]
    assert slipped["most"] > 5 * max(off["most"], 0.1), f"a video one frame off reads {slipped}, no further than the right one"
    MADE["window"].with_suffix(".mp4").unlink()
    return f"frames {want} of the clip differ and are the source under the gate; the laying is {off['most']} levels from its own video at most"


def the_settings() -> str:
    calls = []
    turned = rw.run(args("a", as_rendered=["cuts=none"], set=[f"cuts={FIRST + CUT}"]), decode=handed(calls))
    want = [LEAD + FIRST + f for f in ACROSS]
    assert turned["summary"]["source_frames_differing"] == want and turned["summary"]["of_those_now_the_source"] == want, turned["summary"]
    assert turned["as_rendered"]["cuts"] == [] and turned["changed"]["cuts"] == [FIRST + CUT]
    # --as-rendered alone: the second laying is the window's own settings, not the first laying's
    alone = rw.run(args("a", as_rendered=["cuts=none"]), decode=handed(calls))
    assert alone["changed"]["cuts"] == [FIRST + CUT] and alone["summary"]["of_those_now_the_source"] == want,         f"--as-rendered alone left the second laying at {alone['changed']['cuts']} and moved {alone['summary']['source_frames_differing']}"
    whole = rw.run(args("a", set=["composite=" + vm.COMPOSITE_REGION]), decode=handed(calls))["summary"]
    assert len(whole["source_frames_differing"]) > len(ACROSS) and not set(want) & set(whole["source_frames_differing"]),         f"laid with the whole region the frames that differ are {whole['source_frames_differing']}"
    try:
        rw.change(MADE["settings"], ["grow_pixels=4"])
    except SystemExit:
        pass
    else:
        raise AssertionError("a key the composite does not read was taken as a setting")
    return "the pair turned round names the same frames, now the source; the whole region moves the others; an unknown key is refused"


def the_decode() -> str:
    calls = []
    one = rw.run(args("b", set=["cuts=none"]), decode=handed(calls))
    assert len(calls) == 1 and (WORK / "b" / "decoded_r_window_1.npy").is_file(), f"{len(calls)} decodes for a window with none kept"
    two = rw.run(args("b", set=["cuts=none"]), decode=handed(calls))
    assert len(calls) == 1, "a window with its decode kept was decoded again"
    assert one["summary"] == two["summary"] and "read from" in two["inputs"]["decode"], two["inputs"]
    return "decoded once, kept, read the second time; the same rows both times"


def a_box() -> str:
    import csv
    rw.run(args("f", box="20,36,18,30"), decode=handed([]))          # the box is the bright block the decode draws
    rows = {int(r["frame"]): r for r in csv.DictReader(open(WORK / "f" / "r_window_1.csv"))}
    for f in ACROSS:
        r = rows[f]
        assert float(r["box_decode_off_source"]) > 50 and float(r["box_laid_off_source"]) == 0 and float(r["box_share_kept"]) == 0,             f"frame {f} across the cut: the decode drew the block and the laying must have given the source back, got {r}"
        assert int(r["step_split_by_a_cut"]) == 1 and int(r["step"]) == STEP, r
    before = rows[CUT - 1]
    assert float(before["box_laid_off_source"]) == float(before["box_decode_off_source"]) > 50 and float(before["box_share_kept"]) == 1,         f"a frame the subject is on: the box must be the decode's own, got {before}"
    plain = rows[STARTS[STEP] - 1]
    assert int(plain["step_split_by_a_cut"]) == 0 and float(plain["detail_render"]) > 0 and plain["moved_source"] not in ("", None), plain
    return "across the cut the box is drawn by the decode and given back by the laying; on the subject's frames it is the decode's; the step columns say which step is split"


def a_capture(folder: Path, shift: int = 0, **run_more) -> None:
    """A capture of the made run, as `bench/capture_masked_run.py` lays one out: the mask and the region it read back."""
    (folder / "runs" / "r").mkdir(parents=True)
    lead = 2                                         # the capture starts two frames before the load
    carried = np.zeros((lead + FRAMES, H, W), np.uint8)
    carried[lead:] = MADE["mask"].numpy().astype(np.uint8)
    region = np.zeros((lead + FRAMES, LATENT[3], LATENT[4]), bool)
    region[lead:] = (vm.pixel_alpha(MADE["tokens"], LATENT[3], LATENT[4], 0) > 0.5).numpy()
    if shift:
        region[lead:, 0, 0] ^= True
    np.savez(folder / "runs" / "r" / "region.npz", carried=np.packbits(carried, axis=-1), region=region)
    (folder / "manifest.json").write_text(json.dumps({
        "first_frame": LEAD - lead, "size": [W, H],
        "runs": [{"name": "r", "first_source_frame": LEAD, "others": [], "keep": [], **run_more}]}))
    (folder / "runs" / "r" / "graph.json").write_text(json.dumps({"7": {"class_type": "MiniMaxH3MaskedSource", "inputs": {
        "grow_pixels": MARGIN, "grow_by": vm.GROW_FIXED, "feather_pixels": FEATHER, "composite": vm.COMPOSITE_CHANGED,
        "change_threshold": vm.CHANGE_THRESHOLD, "replace": vm.REPLACE_WHOLE, "edge": vm.EDGE_TOKENS}}}))


def rebuilt() -> str:
    calls = []
    bare = WORK / "bare_windows"
    bare.mkdir()
    window = bare / "r_window_1.safetensors"
    shutil.copy(MADE["window"], window)             # the same stored latent, with no region beside it
    (WORK / "shots.json").write_text(json.dumps({"cuts": [FIRST + CUT]}))
    a_capture(WORK / "cap")
    common = dict(window=str(window), run="r", shots=str(WORK / "shots.json"), context=TRIM, set=["cuts=none"])
    got = rw.run(args("c", capture=str(WORK / "cap"), **common), decode=handed(calls))
    saved = rw.run(args("a", set=["cuts=none"]), decode=handed(calls))
    assert got["summary"] == saved["summary"], f"a rebuilt region names {got['summary']}, the saved one {saved['summary']}"
    assert got["window_first_frame"] == FIRST and got["inputs"]["cells_differing"] == 0, got["inputs"]
    a_capture(WORK / "cap_off", shift=1)
    off = rw.run(args("d", capture=str(WORK / "cap_off"), **common), decode=handed(calls))["inputs"]
    assert off["cells_differing"] == FRAMES, f"a region one cell off on every frame reads {off['cells_differing']} cells"
    a_capture(WORK / "cap_keep", keep=["someone"])
    for bad, why in ((dict(capture=str(WORK / "cap_keep"), **common), "a run that wires keep"),
                     (dict(window=str(window)), "a window with no region and no capture"),
                     ({**common, "capture": str(WORK / "cap"), "context": None}, "a rebuilt region with no first frame")):
        try:
            rw.run(args("e", **bad), decode=handed(calls))
        except SystemExit:
            continue
        raise AssertionError(f"{why} was not refused")
    return "the same frames as the saved region, no cell differing; a cell off a frame is counted; keep, no region and no first frame refused"


case("a window with its saved region", saved_region)
case("the settings of each laying", the_settings)
case("the decode is kept", the_decode)
case("a box before and after the composite", a_box)
case("a region rebuilt from a capture", rebuilt)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(finish())
