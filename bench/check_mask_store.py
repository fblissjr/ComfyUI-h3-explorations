#!/usr/bin/env python3
"""A kept subject mask is found when nothing that decides it has changed, and never otherwise.

`mask_store.py` keeps `MiniMaxH3MaskedSource`'s finished mask on disk so a
restart does not pay for the tracker and the part detection again. The danger
of any such store is a stale hit: a mask from other settings or another video
used in silence, on a render that still plays. Each item is one way that
could happen, or one way the saving could quietly not happen.

1. **The key moves with everything that changes the mask**: a setting on an
   upstream node (the tracker's threshold, the phrase, the object indices),
   the node's own `replace` and `part_*` settings, the frames' content, and
   the size or time of an input file an upstream node names, and the
   `MASK_VERSION` of an upstream node's class, which is how a change to the
   code that makes a mask reaches the key. A class with no version adds
   nothing.
2. **The key does not move with what acts after the mask**: every name in
   `video_mask.MASK_KEY_SKIP`. And that list is exactly the node's inputs
   that `_settle_mask` never reads, so a new input cannot be left out of the
   key by forgetting it.
3. **A kept mask is the tracked one's bytes**, soft edges included, and a
   mask of another shape under the same key is not returned.
4. **The store stays inside its budget**, least recently used out first, a
   read counting as use. RED CONTROL: with room for all, nothing is removed.
   It lives in `masks/` of the output folder, a file carries its source
   video's name, and a file it did not write there is never counted or
   removed.
5. **On a hit core is asked for nothing**: `check_lazy_status` returns no
   input, `execute` runs with no mask and no segmenter, never calls the part
   detection, and returns what the tracked run returned. On a miss it asks
   for `mask` alone, or for `mask` and the segmenter pair under
   `head and hair`.
6. **RED CONTROL: `reuse_mask` off asks every time and reads nothing**, even
   with a matching mask on disk; and with no queued prompt (a direct call)
   nothing is kept or read.
7. **A kept mask that does not read is removed and said so**, not replaced
   by an empty one.
8. **The shipped graphs write `reuse_mask`**, at `h3_config.MASKED_SOURCE`'s
   value.

No model, no CUDA, no server; the store is a temporary folder.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_mask_store.py
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
WORKFLOWS = REPO / "workflows"
sys.path.insert(0, str(WORKFLOWS))
sys.path.insert(0, str(COMFY))

import torch  # noqa: E402

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True  # no CUDA context; the sibling checks do the same
import folder_paths  # noqa: E402
import h3_config  # noqa: E402

# the pack's modules as members of a stand-in package (`check_audio_freeze.py` says why)
pkg = types.ModuleType("_h3pack")
pkg.__path__ = [str(REPO)]
sys.modules.setdefault("_h3pack", pkg)
vm = importlib.import_module("_h3pack.video_mask")
#: The pack ships with kept masks disabled in code (`video_mask.MASK_REUSE_ENABLED`, 2026-10-07). The store's own
#: behaviour is still checked here, as it will be when the switch is turned back on, so the cases below run with
#: it on; `check_disabled` is the case for the shipped state.
SHIPPED_ENABLED = vm.MASK_REUSE_ENABLED
vm.MASK_REUSE_ENABLED = True
ms = importlib.import_module("_h3pack.mask_store")

NODE = "104"
N, H, W = 9, 32, 48


def _prompt(video="clip.mp4"):
    """The shape of the shipped masked graph around the Masked Source, by class and input name."""
    return {
        "28": {"class_type": "VHS_LoadVideoFFmpeg", "inputs": {"video": video, "force_rate": 24.0, "frame_load_cap": 768}},
        "100": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sam.safetensors"}},
        "101": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["100", 1], "text": "man"}},
        "102": {"class_type": "SAM3_VideoTrack", "inputs": {"images": ["28", 0], "model": ["100", 0],
                                                            "conditioning": ["101", 0], "detection_threshold": 0.5,
                                                            "max_objects": 4, "detect_interval": 1}},
        "103": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["102", 0], "object_indices": ""}},
        NODE: {"class_type": "MiniMaxH3MaskedSource",
               "inputs": {"frames": ["28", 0], "mask": ["103", 0], "segmenter": ["100", 0],
                          # `reuse_mask` on, which the shipped config no longer is: this file checks the store
                          "segmenter_clip": ["100", 1], **{**h3_config.MASKED_SOURCE, "reuse_mask": True}}},
    }


def _frames(seed=0):
    return torch.rand((N, H, W, 3), generator=torch.Generator().manual_seed(seed))


def _mask():
    m = torch.zeros((N, H, W))
    m[:, 8:24, 16:32] = 1.0
    m[:, 7, 16:32] = 0.37            # a soft edge, as SAM's resize leaves
    m[0] = 0.0                       # a frame with nobody in it
    return m


def _node(prompt):
    """The node as core calls it: a class clone carrying the hidden inputs."""
    return type("Clone", (vm.MiniMaxH3MaskedSource,), {"hidden": types.SimpleNamespace(prompt=prompt, unique_id=NODE)})


def _args(out):
    return getattr(out, "args", out)


def check_key(problems):
    frames = _frames()
    base = ms.mask_key(_prompt(), NODE, frames, vm.MASK_KEY_SKIP)
    if base is None or base != ms.mask_key(_prompt(), NODE, frames, vm.MASK_KEY_SKIP):
        problems.append("the same prompt and frames do not give the same key")

    def changed(nid, name, value):
        p = _prompt()
        p[nid]["inputs"][name] = value
        return ms.mask_key(p, NODE, frames, vm.MASK_KEY_SKIP) != base

    for nid, name, value in (("102", "detection_threshold", 0.7), ("101", "text", "lead singer"),
                             ("103", "object_indices", "4,6,10"), ("102", "max_objects", 16),
                             ("28", "frame_load_cap", 810), (NODE, "replace", vm.REPLACE_PART),
                             (NODE, "part_phrases", "hair"), (NODE, "part_threshold", 0.3),
                             (NODE, "part_margin", 16)):
        if not changed(nid, name, value):
            problems.append(f"the key does not move when {name} changes on node {nid}")
    if ms.mask_key(_prompt(), NODE, _frames(seed=1), vm.MASK_KEY_SKIP) == base:
        problems.append("the key does not move when the frames differ")
    other = _frames().clone()
    other[5, ::ms.FINGERPRINT_STRIDE, ::ms.FINGERPRINT_STRIDE] += 0.25
    if ms.mask_key(_prompt(), NODE, other, vm.MASK_KEY_SKIP) == base:
        problems.append("the key does not move when one frame of the same shape differs")
    for name, value in (("grow_pixels", 64), ("feather_pixels", 4), ("paint_out", True),
                        ("composite", vm.COMPOSITE_CHANGED), ("change_threshold", 0.2), ("reuse_mask", False)):
        if name not in vm.MASK_KEY_SKIP:
            problems.append(f"{name} acts after the mask and is not in MASK_KEY_SKIP")
        elif changed(NODE, name, value):
            problems.append(f"the key moves with {name}, which does not change the mask")
    # the skip list against the code: the inputs `_settle_mask` takes are the ones that change the mask
    settles = set(inspect.signature(vm.MiniMaxH3MaskedSource._settle_mask).parameters)
    inputs = [i.id for i in vm.MiniMaxH3MaskedSource.define_schema().inputs]
    for name in inputs:
        if (name in settles) == (name in vm.MASK_KEY_SKIP):
            problems.append(f"`{name}`: read by _settle_mask is {name in settles}, in MASK_KEY_SKIP is "
                            f"{name in vm.MASK_KEY_SKIP}; an input is in the key exactly when it settles the mask")
    # an input file's size and time
    with tempfile.TemporaryDirectory() as d:
        real_exists, real_path = folder_paths.exists_annotated_filepath, folder_paths.get_annotated_filepath
        f = Path(d) / "clip.mp4"
        f.write_bytes(b"one")
        folder_paths.exists_annotated_filepath = lambda name: name == "clip.mp4"
        folder_paths.get_annotated_filepath = lambda name, default_dir=None: str(f)
        try:
            a = ms.mask_key(_prompt(), NODE, frames, vm.MASK_KEY_SKIP)
            f.write_bytes(b"another file")
            b = ms.mask_key(_prompt(), NODE, frames, vm.MASK_KEY_SKIP)
        finally:
            folder_paths.exists_annotated_filepath, folder_paths.get_annotated_filepath = real_exists, real_path
        if a == b or a == base:
            problems.append("the key does not move when an input file an upstream node names is replaced")
    # the code that makes the mask: a class's MASK_VERSION, read from the node registry
    def with_classes(**versions):
        classes = {name: type(name, (), {"MASK_VERSION": v} if v is not None else {}) for name, v in versions.items()}
        return ms.mask_key(_prompt(), NODE, frames, vm.MASK_KEY_SKIP, classes=classes)
    v1 = with_classes(SAM3_VideoTrack=None, MiniMaxH3MaskedSource=1)
    if v1 == with_classes(SAM3_VideoTrack=None, MiniMaxH3MaskedSource=2):
        problems.append("the key does not move when an upstream node's MASK_VERSION changes")
    if with_classes(SAM3_VideoTrack=None) != base or with_classes() != base:
        problems.append("a class with no MASK_VERSION, or no registry, changes the key")
    if v1 == base:
        problems.append("declaring a MASK_VERSION does not enter the key")
    if with_classes(SomeOtherNode=7) != base:
        problems.append("the MASK_VERSION of a class that is not upstream entered the key")
    if ms.mask_key(None, NODE, frames) is not None or ms.mask_key(_prompt(), "999", frames) is not None:
        problems.append("a missing prompt or node gives a key; nothing should be kept without one")


def check_store(problems, real_root):
    import folder_paths as fp
    if real_root() != Path(fp.get_output_directory()) / "masks":
        problems.append(f"kept masks go to {real_root()}, not `masks/` in the output folder")
    mask = _mask()
    k1, a, b, c, absent = ("1" * 40, "a" * 40, "b" * 40, "c" * 40, "d" * 40)
    ms.save(k1, mask)
    back = ms.load(k1, (N, H, W))
    if back is None or back.dtype != torch.float32 or not torch.equal(back, mask):
        problems.append("a kept mask is not the tracked mask's bytes")
    if ms.load(k1, (N + 1, H, W)) is not None or ms.has(k1, (N, H, W + 1)):
        problems.append("a kept mask of another shape is returned")
    if not ms.has(k1, (N, H, W)) or ms.has(absent, (N, H, W)) or ms.load(None, (N, H, W)) is not None:
        problems.append("`has` and `load` disagree with what is on disk")
    # the shot table rides in the mask's file: kept without one, a table is not found; kept with one, it is the same text
    if ms.has(k1, (N, H, W), with_table=True) or ms.table(k1) != "" or ms.table(absent) != "" or ms.table(None) != "":
        problems.append("a mask kept without a shot table is reported as carrying one")
    text = '{"table": "h3 shot table", "note": "caf\\u00e9 \\u2014 two shots"}'
    ms.save(k1, mask, text)
    if not ms.has(k1, (N, H, W), with_table=True) or ms.table(k1) != text:
        problems.append("a shot table kept with a mask does not read back as the same text")
    if not torch.equal(ms.load(k1, (N, H, W)), mask) or ms.has(k1, (N + 1, H, W), with_table=True):
        problems.append("keeping a shot table changed the mask, or a table is found under another shape")
    # a file is named after its source video and found again by its key alone
    frames = _frames()
    real_exists, real_path = folder_paths.exists_annotated_filepath, folder_paths.get_annotated_filepath
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "clip.mp4"
        f.write_bytes(b"x")
        folder_paths.exists_annotated_filepath = lambda name: name == "my clip (1).mp4"
        folder_paths.get_annotated_filepath = lambda name, default_dir=None: str(f)
        try:
            key = ms.mask_key(_prompt(video="my clip (1).mp4"), NODE, frames, vm.MASK_KEY_SKIP)
        finally:
            folder_paths.exists_annotated_filepath, folder_paths.get_annotated_filepath = real_exists, real_path
    ms.save(key, mask)
    name = ms._path(key).name
    if name != f"my-clip-1_{key}.npz":
        problems.append(f"a kept mask's file is named {name}, not after its source video and its key")
    ms._LABELS.clear()                                  # another process: the label is not known, the key still finds it
    if not ms.has(key, (N, H, W)) or ms.load(key, (N, H, W)) is None:
        problems.append("a kept mask is not found by its key once its label is forgotten")
    ms._path(key).unlink()
    # the budget: three files, the oldest read last, so the middle one is least recently used;
    # and a file this module did not write, which must not be counted or removed
    stranger = ms.root() / "someone_elses.npz"
    stranger.write_bytes(b"0" * 4096)
    os.utime(stranger, (time.time() - 1000, time.time() - 1000))
    for i, k in enumerate((a, b, c)):
        ms.save(k, mask)
        os.utime(ms._path(k), (time.time() - 100 + i, time.time() - 100 + i))
    ms.load(a, (N, H, W))
    size = ms._path(a).stat().st_size
    ms._path(k1).unlink()
    if ms.evict(budget=10 * size):
        problems.append("RED CONTROL failed: with room for all three something was removed")
    gone = ms.evict(budget=2 * size)
    if gone != [f"{b}.npz"]:
        problems.append(f"past the budget the least recently used should go (b), got {gone}")
    if not stranger.exists():
        problems.append("a file this module did not write was removed from the masks folder")
    stranger.unlink(missing_ok=True)
    for k in (a, c):
        ms._path(k).unlink(missing_ok=True)


def check_node(problems):
    frames, mask = _frames(), _mask()
    for replace in (vm.REPLACE_WHOLE, vm.REPLACE_PART):
        prompt = _prompt()
        prompt[NODE]["inputs"]["replace"] = replace
        node = _node(prompt)
        settings = {k: v for k, v in prompt[NODE]["inputs"].items() if not isinstance(v, list)}
        asks = node.check_lazy_status(frames=frames, mask=None, segmenter=None, segmenter_clip=None, **settings)
        want = ["mask"] if replace == vm.REPLACE_WHOLE else ["mask", "segmenter", "segmenter_clip"]
        if sorted(asks) != sorted(want):
            problems.append(f"{replace}, nothing kept: core is asked for {asks}, expected {want}")
        calls = []
        real = vm.detect_part
        vm.detect_part = lambda seg, clip, fr, where, phrases, threshold=0.5: (calls.append(1), mask[where])[1]
        try:
            tracked = _args(node.execute(frames, mask, segmenter=object(), segmenter_clip=object(), **settings))
            if replace == vm.REPLACE_PART and not calls:
                problems.append("the part detection did not run on a miss; this case tests nothing")
            calls.clear()
            asks = node.check_lazy_status(frames=frames, mask=None, segmenter=None, segmenter_clip=None, **settings)
            if asks:
                problems.append(f"{replace}, a mask kept: core is still asked for {asks}")
            kept = _args(node.execute(frames, None, **settings))
            if calls:
                problems.append(f"{replace}: the part detection ran on a hit")
        finally:
            vm.detect_part = real
        if not torch.equal(kept[1], tracked[1]) or not torch.equal(kept[0]["mask"], tracked[0]["mask"]):
            problems.append(f"{replace}: the run on a kept mask does not return the tracked run's mask")
        tensors = ("frames", "mask", "subject_boxes")
        if {k: v for k, v in kept[0].items() if k not in tensors} != {k: v for k, v in tracked[0].items() if k not in tensors}:
            problems.append(f"{replace}: the source bundle differs between a kept and a tracked run")
        # the subject's boxes: the tracker's on a tracked run; on a kept run the tracker did not run, so they
        # are the kept region's, which is the same thing only when the whole subject is replaced
        if not torch.equal(tracked[0]["subject_boxes"], vm._tracked_boxes(mask)):
            problems.append(f"{replace}: a tracked run's `subject_boxes` are not the tracker's mask's")
        if not torch.equal(kept[0]["subject_boxes"], vm._tracked_boxes(kept[0]["mask"])):
            problems.append(f"{replace}: a kept run's `subject_boxes` are not the kept region's")
        # a setting that acts after the mask keeps the hit; one that changes it does not
        later = dict(settings, grow_pixels=64, composite=vm.COMPOSITE_CHANGED)
        p2 = copy.deepcopy(prompt)
        p2[NODE]["inputs"].update(grow_pixels=64, composite=vm.COMPOSITE_CHANGED)
        if _node(p2).check_lazy_status(frames=frames, mask=None, segmenter=None, segmenter_clip=None, **later):
            problems.append(f"{replace}: changing grow_pixels and composite tracked again")
        p3 = copy.deepcopy(prompt)
        p3["102"]["inputs"]["detection_threshold"] = 0.9
        if not _node(p3).check_lazy_status(frames=frames, mask=None, segmenter=None, segmenter_clip=None, **settings):
            problems.append(f"{replace}: a changed tracker threshold used the old mask")
        # the shot table: wiring it does not change the mask's key, a kept mask without one is a miss once,
        # and after that a hit yields both with nothing asked
        p4 = copy.deepcopy(prompt)
        p4[NODE]["inputs"]["shot_table"] = ["103", 1]
        wired = _node(p4)
        if wired._mask_key(frames, True) != node._mask_key(frames, True):
            problems.append(f"{replace}: wiring the shot table changed the kept mask's key")
        if kept[0].get("shot_table") != "":
            problems.append(f"{replace}: a source with no table wired carries {kept[0].get('shot_table')!r}")
        asks = wired.check_lazy_status(frames=frames, mask=None, segmenter=None, segmenter_clip=None,
                                       shot_table=None, **settings)
        if "shot_table" not in asks or "mask" not in asks:
            problems.append(f"{replace}: a kept mask with no table and the table wired asks for {asks}, "
                            "not the mask and the table")
        text = '{"table": "h3 shot table", "shots": []}'
        vm.detect_part = lambda seg, clip, fr, where, phrases, threshold=0.5: mask[where]
        try:
            first = _args(wired.execute(frames, mask, segmenter=object(), segmenter_clip=object(),
                                        shot_table=text, **settings))
            asks = wired.check_lazy_status(frames=frames, mask=None, segmenter=None, segmenter_clip=None,
                                           shot_table=None, **settings)
            if asks:
                problems.append(f"{replace}: a mask kept with its table: core is still asked for {asks}")
            again = _args(wired.execute(frames, None, **settings))
        finally:
            vm.detect_part = real
        if first[0].get("shot_table") != text or again[0].get("shot_table") != text:
            problems.append(f"{replace}: the table is not the tracker's on the tracked run and on the kept one")
        if not torch.equal(again[1], tracked[1]):
            problems.append(f"{replace}: keeping the table changed the kept mask")
        # the table is the tracker's own: a changed tracker setting is a new key, with no table under it
        p5 = copy.deepcopy(p4)
        p5["102"]["inputs"]["detection_threshold"] = 0.9
        if "shot_table" not in _node(p5).check_lazy_status(frames=frames, mask=None, segmenter=None,
                                                           segmenter_clip=None, shot_table=None, **settings):
            problems.append(f"{replace}: a changed tracker threshold reused the old shot table")
        # RED CONTROL: turned off, it asks and reads nothing
        off = dict(settings, reuse_mask=False)
        if "mask" not in node.check_lazy_status(frames=frames, mask=None, segmenter=None, segmenter_clip=None, **off):
            problems.append(f"RED CONTROL failed: {replace} with reuse_mask off did not ask for the mask")
    # no queued prompt: nothing kept, nothing read
    before = sorted(p.name for p in ms.root().glob("*.npz"))
    plain = vm.MiniMaxH3MaskedSource
    _args(plain.execute(_frames(seed=5), mask))
    if sorted(p.name for p in ms.root().glob("*.npz")) != before:
        problems.append("a direct call with no queued prompt kept a mask")
    if "mask" not in plain.check_lazy_status(frames=_frames(seed=5), mask=None):
        problems.append("a direct call with no queued prompt did not ask for the mask")
    # a kept file that does not read
    prompt = _prompt()
    node = _node(prompt)
    settings = {k: v for k, v in prompt[NODE]["inputs"].items() if not isinstance(v, list)}
    key = node._mask_key(frames, True)
    ms._path(key).write_bytes(b"not a mask")
    if not node.check_lazy_status(frames=frames, mask=None, **settings):
        problems.append("a kept file that does not read still counts as a hit")
    ms.save(key, mask)
    ms._path(key).write_bytes(ms._path(key).read_bytes()[:200])     # the header reads, the body does not
    try:
        node.execute(frames, None, **settings)
        problems.append("a truncated kept mask was used")
    except ValueError as exc:
        if "queue" not in str(exc) or ms._path(key).exists():
            problems.append(f"a truncated kept mask is not removed with a message to queue again: {exc}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"a truncated kept mask raised {type(exc).__name__}, not the node's own message: {exc}")
    schema = vm.MiniMaxH3MaskedSource.define_schema()
    ids = [i.id for i in schema.inputs]
    switch = schema.inputs[ids.index("reuse_mask")] if "reuse_mask" in ids else None
    if switch is None or switch.default is not False or not switch.optional:
        problems.append("reuse_mask is not an optional input, off by default (kept masks are disabled in code since "
                        "2026-10-07; the default goes back on with `video_mask.MASK_REUSE_ENABLED`)")
    # Inputs appended after the switch (the motion reference, 2026-10-05) must be
    # optional, so a saved graph keeps running, and must not reach the kept
    # mask's key, since they do not change the mask.
    after = [i for i in schema.inputs[ids.index("reuse_mask") + 1:]] if switch is not None else []
    for i in after:
        if not i.optional:
            problems.append(f"{i.id} is appended after reuse_mask and is not optional")
        if i.id not in vm.MASK_KEY_SKIP:
            problems.append(f"{i.id} is appended after reuse_mask and would enter the kept mask's key")
    lazy = sorted(i.id for i in schema.inputs if getattr(i, "lazy", False))
    expected_lazy = sorted(vm.LAZY_FOR_MASK + (vm.LAZY_FOR_TABLE, vm.LAZY_FOR_PARTS))
    if lazy != expected_lazy:
        problems.append(f"the lazy inputs are {lazy}, expected {expected_lazy}; `frames` must not be lazy")


def check_disabled(problems):
    """As shipped: no key, so nothing is read and nothing is written, whatever the node's `reuse_mask` says."""
    if SHIPPED_ENABLED is not False:
        problems.append(f"video_mask.MASK_REUSE_ENABLED ships as {SHIPPED_ENABLED!r}; it is False until the masked lane is "
                        "ready for production, by the owner's decision of 2026-10-07, and is turned on by a code change")
    node, frames = _node(_prompt()), _frames()
    on = node._mask_key(frames, True)
    vm.MASK_REUSE_ENABLED = False
    try:
        off = node._mask_key(frames, True)
    finally:
        vm.MASK_REUSE_ENABLED = True
    if on is None or off is not None:
        problems.append(f"with the switch on the key is {on!r} and with it off {off!r}: off must give no key with "
                        "`reuse_mask` on, and on must give one, or the switch is not what gates the store")
    text = next((str(i.tooltip) for i in vm.MiniMaxH3MaskedSource.define_schema().inputs if i.id == "reuse_mask"), "")
    if not SHIPPED_ENABLED and "no effect" not in text:
        problems.append("`reuse_mask`'s tooltip does not say that it has no effect while kept masks are disabled in code")
    # the other result kept on disk, a rendered window, is gated the same way (`audio_freeze_song.WINDOW_REUSE_ENABLED`)
    song = importlib.import_module("_h3pack.audio_freeze_song")
    if song.WINDOW_REUSE_ENABLED is not False:
        problems.append(f"audio_freeze_song.WINDOW_REUSE_ENABLED ships as {song.WINDOW_REUSE_ENABLED!r}; it is False until a "
                        "stored window's key changes when the code does (the owner, 2026-10-07)")
    source = Path(song.__file__).read_text()
    if "if WINDOW_REUSE_ENABLED and reuse_windows and root is not None:" not in source:
        problems.append("the song node's reuse of stored windows is not gated by WINDOW_REUSE_ENABLED")


def check_graphs(problems):
    seen = 0
    for path in h3_config.graph_paths(WORKFLOWS, include_bench=True):
        graph = json.loads(Path(path).read_text())
        if not isinstance(graph, dict):
            continue
        for node in graph.values():
            if isinstance(node, dict) and node.get("class_type") == "MiniMaxH3MaskedSource":
                seen += 1
                if node["inputs"].get("reuse_mask") is not h3_config.MASKED_SOURCE["reuse_mask"]:
                    problems.append(f"{Path(path).name}: reuse_mask is {node['inputs'].get('reuse_mask')!r}, "
                                    f"not h3_config.MASKED_SOURCE's")
    if not seen:
        problems.append("no shipped graph wires a Masked Source; item 8 checked nothing")


def main() -> int:
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as d:
        real = ms.root
        ms.root = lambda: Path(d)
        try:
            check_key(problems)
            check_store(problems, real)
            check_node(problems)
            check_disabled(problems)
        finally:
            ms.root = real
    check_graphs(problems)
    for p in problems:
        print(f"FAIL  {p}")
    if not problems:
        print("ok    a kept mask is found only when nothing that decides it changed, is the tracked mask's "
              "bytes, stays inside its budget, and on a hit core is asked for nothing")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
