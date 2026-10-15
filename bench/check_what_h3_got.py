#!/usr/bin/env python3
"""What a load was handed (`bench/what_h3_got.py`), on a job folder whose every file is made here and known.

Small files are made with ffmpeg and Pillow in a temporary folder: a source, three masks, two reference videos and
two stills, and a graph in ComfyUI's API form that wires them as the job builder does (masks through ImageToMask
and a MaskComposite, two stills chained through `references`, the sound trimmed from the source).

1. **Each input is read by its place in the graph**: the source, the mask, both keep-outs above the composite,
   `<Video 1>` and `<Video 2>` by the Masked Source's input, `<Picture 1>` the START of the stills chain and
   `<Picture 2>` its end, a still named relative to the input folder found there; each digest the file's own; a
   loader's first frame from its start time set a fraction of a frame early; the text the song node's prompt; a
   reference's own record read beside it; the node's words routed to the tag they name.
2. **The loader's cut is VideoHelperSuite's own**: the filter string its ffmpeg path adds, run by ffmpeg on frames
   whose every column says its x, cuts what `vhs_fit` says it cuts, for a 1424x1080 and a 1920x1080 file into
   1024x768, and nothing from a 1024x768 one (the control: the loader's test passes there and the cut is empty).
3. **The hard failures**: a file the graph names that is not there; a file written after `queued_at`; a digest
   recorded at queue time that is not the file's. The control is the same folder with none of the three, which
   reads clean. The digests the queue step keeps (`digests_of`) read back clean, and a file given other bytes
   under its old time fails on the digest alone.
   A file written inside the second the record cuts away is not late; 1.5 s after is.
4. **What a cold read found missing** (2026-10-15): a sound loaded from its own file (`VHS_LoadAudio`) with the
   trim the graph puts on it; the latent a load continues from (`continue_from`), digested like a file; a node
   that is no known loader listed with no file; `parts` wired and not asked for under `replace` = whole subject;
   a graph that appends a reference video refused. The input folder: none known refuses; the queued record's
   `input_dir` is used. ffmpeg's crop on a yuv420p file starts on an even row or column, and `vhs_fit` says so.
5. **Which reference frames the text encoder is shown**: the window cut to 17k+5, every twelfth frame, the
   reference's last frame repeated past its end; the windows read from the node's own lines with no rendered
   window, and the node's own count kept beside.
6. **A load not yet queued** reads as "built, not queued", with no file held to a time.
7. **The zoom**: the node's `_zoomed.json` beside the render the queued record names, and each window's region
   meta, read.
8. **A plan's masks rebuilt in place** (`load_chain`'s on-plan step, 2026-10-15): the plan queued with its digests,
   the masks' frame hashes recorded after it (`<load>.plan.masks.json`, by `load_chain.frame_hash`), a copy of them
   kept (`<capture>_first`), and the masks written again with new bytes and the same frames: the plan reads clean,
   each mask "frames as planned", where until then each failed twice. A mask whose frame 3 alone was changed fails and
   names frame 3, though the step's record were made to agree with it: the copy is the anchor, not the record. With no copy, and with a mask changed between the
   queue and the copy (the copy is then not the queue's bytes), the two rules stand. The render stage is held as
   before.

Needs ffmpeg and ffprobe, Pillow and numpy. No model, no card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_what_h3_got.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402
from _lib.frames import LOADER_CROP  # noqa: E402

import what_h3_got as w  # noqa: E402

HOLD = tempfile.TemporaryDirectory(prefix="check_what_h3_got_")
ROOT = Path(HOLD.name)
QUEUED_AT = "2026-10-15T10:00:00-05:00"
BEFORE = datetime.datetime.fromisoformat("2026-10-15T09:00:00-05:00").timestamp()
AFTER = datetime.datetime.fromisoformat("2026-10-15T11:00:00-05:00").timestamp()


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def clip(path: Path, size: str, frames: int = 6) -> Path:
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={size}:rate=24", "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path))
    return path


def vhs(video: str, at: float = 0.0) -> dict:
    return {"class_type": w.VIDEO_LOADER, "inputs": {"video": video, "custom_width": 1024, "custom_height": 768, "frame_load_cap": 6,
                                                     "start_time": at, "force_rate": 0.0, "format": "AnimateDiff"}}


def made(name: str = "load_a", queued: bool = True) -> Path:
    """One job folder with its graph; the files under ROOT/<name>. Every file written an hour before the queue."""
    base = ROOT / name
    inputs, out, job = base / "input", base / "output", base / f"{name}_job"
    for d in (inputs / "sub", out / "Video" / name, job):
        d.mkdir(parents=True, exist_ok=True)
    src = clip(base / "source.mp4", "1424x1080", 30)
    masks = [clip(base / f"m{k}.mp4", "1024x768") for k in range(3)]
    refs = [clip(base / f"ref{k}.mp4", "1024x768") for k in (1, 2)]
    (base / "ref1.json").write_text(json.dumps({"what": "the body model cut to the head", "its_first_frame_is_source_frame": 10}))
    from PIL import Image
    Image.new("RGB", (60, 90), (200, 10, 10)).save(inputs / "sub" / "one.png")
    Image.new("RGB", (80, 80), (10, 200, 10)).save(base / "two.png")
    g = {"28": vhs(str(src), (10 - 0.4) / 24), "146": {"class_type": "TrimAudioDuration", "inputs": {"audio": ["28", 2]}},
         "150": vhs(str(masks[0])), "151": {"class_type": "ImageToMask", "inputs": {"image": ["150", 0]}},
         "170": vhs(str(masks[1])), "171": {"class_type": "ImageToMask", "inputs": {"image": ["170", 0]}},
         "180": vhs(str(masks[2])), "181": {"class_type": "ImageToMask", "inputs": {"image": ["180", 0]}},
         "190": {"class_type": "MaskComposite", "inputs": {"destination": ["171", 0], "source": ["181", 0], "operation": "add"}},
         "200": vhs(str(refs[0])), "210": vhs(str(refs[1])),
         "15": {"class_type": "LoadImage", "inputs": {"image": "sub/one.png"}},
         "16": {"class_type": "LoadImage", "inputs": {"image": str(base / "two.png")}},
         "50": {"class_type": "MiniMaxH3AppendRefImage", "inputs": {"image": ["15", 0], "size_policy": "max", "size_policy.dit_short_edge": 2048}},
         "51": {"class_type": "MiniMaxH3AppendRefImage", "inputs": {"image": ["16", 0], "references": ["50", 0], "size_policy": "max"}},
         "104": {"class_type": w.SOURCE, "inputs": {"frames": ["28", 0], "mask": ["151", 0], "others": ["190", 0], "motion_video": ["200", 0],
                                                    "motion_video_2": ["210", 0], "replace": "the wired parts", "part_phrases": "hair",
                                                    "render_region": "zoomed in on what is replaced"}},
         "74": {"class_type": w.SONG, "inputs": {"source": ["104", 0], "references": ["51", 0], "audio": ["146", 0], "prompt": "the text\n<Picture 2> is a line of the text, echoed\n"}}}
    (job / f"{name}.render.json").write_text(json.dumps(g))
    for f in [src, *masks, *refs, inputs / "sub" / "one.png", base / "two.png"]:
        os.utime(f, (BEFORE, BEFORE))
    if queued:
        (job / f"{name}.render.queued.json").write_text(json.dumps({
            "queued_at": QUEUED_AT, "status": "success", "files": [f"Video/{name}/{name}_00001.mp4"],
            "said": {"50": ["<Picture 1>: still, 60x90"], "75": ["[1] <Video 2> costs: shown 4 pictures\nnothing tagged here"],
                     "80": ["the text\n<Picture 2> is a line of the text, echoed"]}}))
        piece = out / "Video" / name / f"{name}_00001.mp4"
        (piece.with_name(f"{name}_00001_zoomed.json")).write_text(json.dumps({"box": [8, 8, 520, 392], "zoom": 2.0, "source_size": [1024, 768], "set_by": "the box"}))
        win = out / "Video" / name / f"{name}_windows"
        win.mkdir(exist_ok=True)
        np.savez(win / f"{name}_window_1_region.npz", meta=np.array(json.dumps({"first_frame": 0, "frames": 6, "zoom": {"box": [8, 8, 520, 392]}})))
    return base


def read(base: Path, **kw) -> dict:
    name = base.name
    return w.read(base / f"{name}_job", name, inputs=base / "input", output=base / "output", **kw)


def each_input_by_its_place() -> str:
    base = made()
    got = read(base)
    by = {(r["role"], r["node"]): r for r in got["inputs"]}
    roles = [(r["role"], r["node"]) for r in got["inputs"]]
    for want in (("the source", "28"), ("the subject's mask", "150"), ("the keep-outs", "170"), ("the keep-outs", "180"),
                 ("<Video 1>", "200"), ("<Video 2>", "210"), ("the sound", "28"), ("<Picture 1>", "15"), ("<Picture 2>", "16")):
        assert want in by, f"{want} not read; read {roles}"
    assert [r["node"] for r in got["inputs"] if r["role"].startswith("<Picture")] == ["15", "16"], "the stills chain read out of order"
    one = by[("<Picture 1>", "15")]
    assert one["file"] == str(base / "input" / "sub" / "one.png") and one["size"] == [60, 90], one
    for r in got["inputs"]:
        assert r["sha256"] == hashlib.sha256(Path(r["file"]).read_bytes()).hexdigest(), f"{r['role']}: digest is not the file's"
    assert by[("the source", "28")]["loader"]["first_frame"] == 10, by[("the source", "28")]["loader"]
    assert got["text"]["text"].startswith("the text\n") and got["text"]["sha256"] == hashlib.sha256(got["text"]["text"].encode()).hexdigest()
    assert by[("<Video 1>", "200")]["record"]["its_first_frame_is_source_frame"] == 10
    assert "record" not in by[("<Video 2>", "210")], "a record read for a reference that has none"
    assert by[("<Video 2>", "210")]["node_said"] == ["[1] <Video 2> costs: shown 4 pictures"], by[("<Video 2>", "210")].get("node_said")
    assert one["node_said"] == ["<Picture 1>: still, 60x90"] and "node_said" not in by[("<Picture 2>", "16")]
    assert got["replaces"]["part_phrases"] == "hair" and not got["failures"], got["failures"]
    return f"{len(got['inputs'])} inputs, each by its place; stills in chain order; digests the files'; first frame 10; the clean control has no failure"


def the_loaders_cut() -> str:
    """The cut VideoHelperSuite's filter makes, by ffmpeg, against `vhs_fit`: frames whose columns and rows say x and y."""
    said = []
    for size in ((1424, 1080), (1920, 1080), (1024, 768)):
        fit = w.vhs_fit(list(size), 1024, 768)
        ar = 1024 / 768
        flt = LOADER_CROP.format(ar=ar)                     # the loader's line (held to its file by bench/check_loader_fit.py)
        got = []
        for axis in ("X", "Y"):
            # a level a pixel, in 16-bit grey so a coordinate up to 1920 fits: the first and last of the cut say where it is
            raw = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"nullsrc=size={size[0]}x{size[1]},format=gray16le,geq=lum='{axis}'",
                                  "-frames:v", "1", "-vf", flt, "-f", "rawvideo", "-pix_fmt", "gray16le", "-"], capture_output=True, check=True).stdout
            got.append(np.frombuffer(raw, "<u2"))
        xs, ys = got
        n = len(xs)
        # the cut's width: the run of the first row's x levels that rises by one
        row0 = xs[: next((i for i in range(1, n) if xs[i] <= xs[i - 1]), n)]
        cw, ch = len(row0), n // len(row0)
        x0, y0 = int(row0[0]), int(ys[0])
        want = None if (cw, ch) == size else [x0, y0, cw, ch]
        assert fit["crop"] == want, f"{size}: vhs_fit says {fit['crop']}, ffmpeg cut {want}"
        assert fit["to"] == [1024, 768]
        said.append(f"{size[0]}x{size[1]}: {want or 'no cut'}")
    return "; ".join(said)


def the_hard_failures() -> str:
    base = made("load_b")
    name = base.name
    os.utime(base / "m1.mp4", (AFTER, AFTER))
    (base / "ref2.mp4").unlink()
    q = base / f"{name}_job" / f"{name}.render.queued.json"
    rec = json.loads(q.read_text())
    rec["digests"] = {str(base / "m0.mp4"): "0" * 64, str(base / "source.mp4"): hashlib.sha256((base / "source.mp4").read_bytes()).hexdigest()}
    q.write_text(json.dumps(rec))
    got = read(base)
    f = got["failures"]
    assert any("m1.mp4" in x and "after the load was queued" in x for x in f), f
    assert any("ref2.mp4" in x and "not there" in x for x in f), f
    assert any("m0.mp4" in x and "recorded at queue time" in x for x in f), f
    assert len(f) == 3, f"a failure beyond the three made: {f}"
    return "a file written after the queue, a missing file and a wrong recorded digest each fail; the right recorded digest does not"


def digests_kept_at_queue() -> str:
    """The queue step's `digests` (`digests_of` on the graph) read back clean, and one file changed after fails by its bytes."""
    base = made("load_e")
    name = base.name
    job = base / f"{name}_job"
    q = job / f"{name}.render.queued.json"
    rec = json.loads(q.read_text())
    rec["digests"] = w.digests_of(json.loads((job / f"{name}.render.json").read_text()), base / "input")
    q.write_text(json.dumps(rec))
    assert str(base / "input" / "sub" / "one.png") in rec["digests"] and len(rec["digests"]) == 8, sorted(rec["digests"])
    clean = read(base)
    assert not clean["failures"] and all(r["sha256_at_queue"] == r["sha256"] for r in clean["inputs"]), clean["failures"]
    clip(base / "ref1.mp4", "1024x768", 7)            # other bytes, written back to the time before the queue
    os.utime(base / "ref1.mp4", (BEFORE, BEFORE))
    f = read(base)["failures"]
    assert len(f) == 1 and "ref1.mp4" in f[0] and "recorded at queue time" in f[0], f
    return "the eight named files kept by path, the source once though two loaders read it; read back clean; a file with other bytes and the old time fails on its digest alone"


def what_the_cold_read_found() -> str:
    """A sound loaded from its own file, a latent continued from, a node that is no known loader, a lazy input the
    node does not ask for, a file written within the queue's own second, and a reference video appended."""
    base = made("load_f")
    name = base.name
    job = base / f"{name}_job"
    gf = job / f"{name}.render.json"
    g = json.loads(gf.read_text())
    (base / "vocals.wav").write_bytes(b"RIFF....WAVE")
    (base / "w1.safetensors").write_bytes(b"latent")
    g["145"] = {"class_type": "VHS_LoadAudio", "inputs": {"audio_file": str(base / "vocals.wav"), "seek_seconds": 2.0, "duration": 1.0}}
    g["146"]["inputs"] = {"audio": ["145", 0], "start_index": 0.0, "duration": 0.9}
    g["74"]["inputs"]["continue_from"] = str(base / "w1.safetensors")
    g["300"] = {"class_type": "SolidMask", "inputs": {"value": 1.0}}
    g["104"]["inputs"]["parts"] = ["300", 0]                      # replace is "the wired parts": asked for
    g["104"]["inputs"]["keep"] = ["181", 0]
    gf.write_text(json.dumps(g))
    for f in (base / "vocals.wav", base / "w1.safetensors"):
        os.utime(f, (BEFORE, BEFORE))
    queued_at = datetime.datetime.fromisoformat(QUEUED_AT).timestamp()
    os.utime(base / "m2.mp4", (queued_at + 0.8, queued_at + 0.8))   # inside the second the record cuts away
    got = read(base)
    by = {r["role"]: r for r in got["inputs"]}
    sound = [r for r in got["inputs"] if r["role"] == "the sound"]
    assert len(sound) == 1 and sound[0]["file"] == str(base / "vocals.wav") and sound[0]["trim"]["duration"] == 0.9, sound
    cont = by["continued from (an earlier window's latent)"]
    assert cont["file"] == str(base / "w1.safetensors") and cont["sha256"] == hashlib.sha256(b"latent").hexdigest(), cont
    assert str(base / "w1.safetensors") in w.digests_of(g, base / "input")
    leaf = by["the parts kept"]
    assert leaf["file"] is None and leaf["exists"] is None and "unread" not in leaf, leaf
    assert by["what is kept the original (`keep`)"]["node"] == "180"
    assert not got["failures"], got["failures"]
    g["104"]["inputs"]["replace"] = "whole subject"                # now `parts` is wired and not asked for
    gf.write_text(json.dumps(g))
    assert "not read" in {r["role"]: r for r in read(base)["inputs"]}["the parts kept"]["unread"]
    os.utime(base / "m2.mp4", (queued_at + 1.5, queued_at + 1.5))
    assert any("m2.mp4" in x for x in read(base)["failures"]), "a file written 1.5 s after the queue did not fail"
    g["400"] = {"class_type": "MiniMaxH3AppendRefVideo", "inputs": {}}
    gf.write_text(json.dumps(g))
    try:
        read(base)
    except SystemExit as e:
        assert "appends a reference video" in str(e), e
    else:
        raise AssertionError("a graph with an appended reference video was read")
    return ("the sound from its own file with its trim; the latent continued from, digested; a leaf no loader listed with no file; "
            "`parts` unread under whole subject; 0.8 s past the queue clean, 1.5 s fails; an appended reference video refused")


def the_input_folder() -> str:
    base = made("load_g")
    name = base.name
    held = os.environ.pop("H3_INPUT_DIR", None)
    try:
        try:
            w.read(base / f"{name}_job", name, output=base / "output")
        except SystemExit as e:
            assert "H3_INPUT_DIR" in str(e), e
        else:
            raise AssertionError("read with no input folder known and none refused")
        q = base / f"{name}_job" / f"{name}.render.queued.json"
        q.write_text(json.dumps({**json.loads(q.read_text()), "input_dir": str(base / "input")}))
        got = w.read(base / f"{name}_job", name, output=base / "output")
        assert {r["role"]: r for r in got["inputs"]}["<Picture 1>"]["exists"], "the recorded input_dir was not used"
    finally:
        if held is not None:
            os.environ["H3_INPUT_DIR"] = held
    return "no input folder known: refused by name; the queued record's input_dir used"


def the_cut_on_a_yuv420p_file() -> str:
    """ffmpeg's crop rounds an offset down to the colour's step on yuv420p: a file whose marked row or column says
    where the cut starts, run through the loader's filter, against `vhs_fit`."""
    said = []
    ar = 1024 / 768
    flt = LOADER_CROP.format(ar=ar)
    for (W, H), axis in (((1438, 1080), "Y"), ((1922, 1080), "X")):
        fit = w.vhs_fit([W, H], 1024, 768, pix_fmt="yuv420p")
        at = fit["crop"][1] if axis == "Y" else fit["crop"][0]
        even, odd = at, at + 1                                      # the even start the tool says, and the odd one beside it
        f = ROOT / f"marked_{W}.mp4"
        ffmpeg("-f", "lavfi", "-i", f"color=black:size={W}x{H}:rate=24,format=yuv420p,geq=lum='if(eq({axis},{even}),250,if(eq({axis},{odd}),120,16))':cb=128:cr=128",
               "-frames:v", "1", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(f))
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(f), "-vf", flt, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                             capture_output=True, check=True).stdout
        cw, ch = fit["crop"][2], fit["crop"][3]
        pic = np.frombuffer(raw, np.uint8)
        assert pic.size == cw * ch, f"{W}x{H}: ffmpeg cut {pic.size} pixels, vhs_fit says {cw}x{ch}"
        pic = pic.reshape(ch, cw)
        first = int(pic[0, 5]) if axis == "Y" else int(pic[5, 0])
        assert first > 200, f"{W}x{H}: the cut does not start on {axis.lower()}={even}: its first {'row' if axis == 'Y' else 'column'} is {first}"
        said.append(f"{W}x{H}: starts on {axis.lower()}={even}, {cw}x{ch}")
    return "; ".join(said)


def the_encoder_frames() -> str:
    """Which reference frames the text encoder is shown: the window cut to 17k+5, every twelfth, the reference's last
    frame repeated past its end; read for each window of a rendered load and set beside the node's own count."""
    e = w.encoder_frames(0, 141, 50, 949)
    assert e["kept_frames"] == 141 and [x["reference_frame"] for x in e["sampled"]] == [0, 12, 24, 36, 48] + [49] * 7, e
    assert sum(x["held"] for x in e["sampled"]) == 7 and e["sampled"][1]["source_frame"] == 961
    assert w.encoder_frames(0, 40, 40, None)["kept_frames"] == 39, "40 frames are not cut to 39 (17*2+5)"
    assert len(w.encoder_frames(0, 39, 37, None)["sampled"]) == 4 and len(w.encoder_frames(0, 158, 154, None)["sampled"]) == 14
    later = w.encoder_frames(90, 39, 100, 0)
    assert [x["reference_frame"] for x in later["sampled"]] == [90, 99, 99, 99] and [x["held"] for x in later["sampled"]] == [False, True, True, True]
    base = made("load_h")
    name = base.name
    q = base / f"{name}_job" / f"{name}.render.queued.json"
    rec = json.loads(q.read_text())
    rec["said"]["75"] = ["[1] window from 0.000s: 39 frames\n[1] <Video 1> costs: the text encoder is shown 4 pictures of 1024x768"]
    q.write_text(json.dumps(rec))
    win = base / "output" / "Video" / name / f"{name}_windows"
    for f in win.glob("*"):
        f.unlink()                                                  # no rendered window: the node's own lines say it
    got = read(base)
    v1 = {r["role"]: r for r in got["inputs"]}["<Video 1>"]
    e = v1["encoder_frames"]
    assert len(e) == 1 and e[0]["window_frames"] == 39 and v1["reference_frames"] == 6, (e, v1.get("reference_frames"))
    assert [x["reference_frame"] for x in e[0]["sampled"]] == [0, 5, 5, 5] and e[0]["node_says"] == 4, e
    assert e[0]["sampled"][0]["source_frame"] == 10, "the window's first frame is not the source loader's"
    return "141 frames of a 50-frame reference: 12 shown, 7 of them its last frame; 40 cut to 39; windows from the node's own lines, its count kept beside"


def not_yet_queued() -> str:
    base = made("load_c", queued=False)
    os.utime(base / "m1.mp4", (AFTER, AFTER))
    got = read(base)
    assert got["state"] == "built, not queued" and got["queued"] is None
    assert all(r["written_after_queue"] is None for r in got["inputs"]) and not got["failures"], got["failures"]
    assert got["zoom"]["piece"] is None and got["zoom"]["render_region"] == "zoomed in on what is replaced"
    return "built, not queued: no file is held to a time, the zoom is the graph's words alone"


def the_zoom() -> str:
    base = made("load_d")
    got = read(base)
    z = got["zoom"]
    assert z["piece"]["box"] == [8, 8, 520, 392] and z["piece"]["zoom"] == 2.0, z
    assert len(z["windows"]) == 1 and z["windows"][0]["zoom"]["box"] == [8, 8, 520, 392], z["windows"]
    return "the node's zoomed record beside the render and the window's region meta read"


def a_plans_masks_rebuilt() -> str:
    from load_chain import frame_hash

    def drawn(m: Path) -> None:
        """Frame 3 alone drawn on; every other frame kept exactly (a lossless encode of the decoded frames)."""
        one = m.with_name("one.mp4")
        ffmpeg("-i", str(m), "-vf", "drawbox=x=0:y=0:w=200:h=200:color=red:t=fill:enable='eq(n,3)'", "-c:v", "libx264",
               "-qp", "0", "-pix_fmt", "yuv420p", str(one))
        one.replace(m)
        os.utime(m, (AFTER, AFTER))

    def planned_world(name: str, change_before_copy: bool = False) -> tuple[Path, Path, list[Path]]:
        """A plan queued with its digests; then, as the on-plan step does: the masks' frame hashes recorded, the capture
        copied to `<capture>_first`, and the masks built again in place with new bytes and the same frames."""
        base = made(name)
        job = base / f"{name}_job"
        g = json.loads((job / f"{name}.render.json").read_text())
        (job / f"{name}.plan.json").write_text(json.dumps(g))
        rec = json.loads((job / f"{name}.render.queued.json").read_text())
        rec["digests"] = w.digests_of(g, base / "input")
        (job / f"{name}.plan.queued.json").write_text(json.dumps(rec))
        masks = [base / f"m{k}.mp4" for k in range(3)]
        if change_before_copy:                              # m1's frame 3 changed after the queue, before the step ran
            drawn(masks[1])
        (job / f"{name}.plan.masks.json").write_text(json.dumps({"at": "2026-10-15T15:00:07Z",
                                                                 "masks": {str(m): frame_hash(str(m)) for m in masks}}))
        copy = base.with_name(base.name + "_first")
        copy.mkdir()
        for m in masks:
            (copy / m.name).write_bytes(m.read_bytes())
        for m in masks:
            again = m.with_name("again.mp4")
            ffmpeg("-i", str(m), "-c", "copy", "-metadata", "title=built again", str(again))
            assert again.read_bytes() != m.read_bytes()
            again.replace(m)
            os.utime(m, (AFTER, AFTER))
        return base, job, masks

    plan = lambda base, job: w.read(job, base.name, "plan", inputs=base / "input", output=base / "output")  # noqa: E731
    base, job, masks = planned_world("load_p")
    got = plan(base, job)
    # THE CASE THAT GOES RED when a mask rebuilt in place with the frames the plan read fails again
    assert not got["failures"], got["failures"]
    said = [r for r in got["inputs"] if r.get("rebuilt")]
    assert len(said) == 3 and all("frames as planned" in r["rebuilt"] for r in said), [r.get("rebuilt") for r in got["inputs"]]
    assert any("frames as planned" in x for x in w.said_lines(got))
    # one frame changed after the rebuild: fails, naming it, though the step's record were made to agree with the
    # changed file: the copy is the anchor, not the record (a cold reader's case)
    drawn(masks[1])
    rec = json.loads((job / "load_p.plan.masks.json").read_text())
    (job / "load_p.plan.masks.json").write_text(json.dumps({**rec, "masks": {**rec["masks"], str(masks[1]): frame_hash(str(masks[1]))}}))
    f = plan(base, job)["failures"]
    assert len(f) == 1 and "m1.mp4" in f[0] and "not what the plan read" in f[0] and "frame 3 " in f[0], f
    # no copy: the two rules stand, and nothing breaks
    base.with_name("load_p_first").rename(base.with_name("load_p_kept_aside"))
    f = plan(base, job)["failures"]
    assert any("m0.mp4" in x and "after the load was queued" in x for x in f), f
    # changed after the queue and before the step ran: the copy is not the queue's bytes, so the two rules stand
    base2, job2, _m = planned_world("load_q", change_before_copy=True)
    f = plan(base2, job2)["failures"]
    assert any("m1.mp4" in x and "after the load was queued" in x for x in f) and not any("m0.mp4" in x for x in f), f
    # the render stage of the same folder: held by bytes and time as before
    f = w.read(job2, "load_q", "render", inputs=base2 / "input", output=base2 / "output")["failures"]
    assert any("m0.mp4" in x and "after the load was queued" in x for x in f), f
    return ("three masks rebuilt in place with the queue's frames read clean (\"frames as planned\"); frame 3 changed fails "
            "naming frame 3, with the step's record agreeing with the changed file; with no copy, and with a mask changed between the queue and the copy, the "
            "two rules stand; the render stage is held as before")


case("each input read by its place in the graph", each_input_by_its_place)
case("the loader's cut, against ffmpeg running the loader's filter", the_loaders_cut)
case("the hard failures, and a clean control", the_hard_failures)
case("digests kept at queue time", digests_kept_at_queue)
case("what a cold read found", what_the_cold_read_found)
case("the input folder", the_input_folder)
case("the cut on a yuv420p file, by ffmpeg", the_cut_on_a_yuv420p_file)
case("which reference frames the text encoder is shown", the_encoder_frames)
case("a load not yet queued", not_yet_queued)
case("the zoom read from the node's records", the_zoom)
case("a plan's masks rebuilt in place", a_plans_masks_rebuilt)

if __name__ == "__main__":
    code = finish()
    HOLD.cleanup()
    sys.exit(code)
