#!/usr/bin/env python3
"""Hold the keep a song window's source latent and conditioning are reused from (`window_keep.py`).

The keep hands a later run a tensor an earlier run made, so every way it could
hand over the wrong one, or one that has changed, is a case here. No card, no
model: the store and its two key builders are plain Python over tensors.

  identity         the same objects hit; a new tensor of equal content misses;
                   a changed plain value misses; an entry goes when an object
                   it was computed from is collected.
  the budget       least recently used out first past the byte budget, a value
                   over the budget alone is not kept, and a host short of
                   memory keeps nothing. RED CONTROL: with room nothing goes.
  on the host      every tensor of a kept value is on the CPU, through the
                   one line that moves it (`_to_host`), shown with a tensor on
                   a device that is not the CPU.
  not written to   a kept value written into after it was kept is dropped at
                   the next `get` and counted, and what `get` returns can be
                   appended to without changing what is kept. RED CONTROL: an
                   untouched value hits.
  the latent key   the mask is in it only under `paint_out` or a late start;
                   the window, the canvas and the VAE's dtypes always are.
  the conditioning the text, the encoder's patches and each reference record
    key            are in it; the frames and the mask only with a motion
                   reference; it builds with no source and no references.
  two restated     `START_NOISE` and `MOTION_NONE` equal `video_mask.py`'s.
    constants
  the song node    `audio_freeze_song.py` reads either keep only under its
                   `reuse_windows` input and stores into both whatever the
                   switch says, as its windows on disk are stored.

    <comfy venv python> bench/check_window_keep.py
"""
from __future__ import annotations

import gc
import importlib.util
import sys
import types
import uuid
from pathlib import Path

from _lib import bootstrap, case, finish

REPO = Path(__file__).resolve().parent.parent


def load(name: str):
    """A root module of the pack as a module of a stand-in package (`check_audio_freeze.py` says why)."""
    pkg = sys.modules.setdefault("_h3pack", types.ModuleType("_h3pack"))
    pkg.__path__ = [str(REPO)]
    spec = importlib.util.spec_from_file_location(f"_h3pack.{name}", REPO / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"_h3pack.{name}"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    bootstrap(cpu=True)             # `video_mask.py` imports core; nothing here uses a device
    import torch

    wk = load("window_keep")
    MiB = 1024 ** 2

    def keep(budget=64 * MiB, short=lambda: False):
        return wk.Keep("test value", budget, short=short)

    class Thing:                    # stands in for a VAE, an encoder or a reference record
        pass

    # ------------------------------------------------------------- identity
    def same_objects_hit():
        k, frames, vae = keep(), torch.rand(4, 8, 8, 3), Thing()
        value = torch.rand(2, 3)
        assert k.put(("a", 1), (frames, vae), value)
        got = k.get(("a", 1), (frames, vae))
        assert got is not None and torch.equal(got, value)
    case("the same plain values and the same objects hit", same_objects_hit)

    def equal_content_misses():
        k, frames, vae = keep(), torch.rand(4, 8, 8, 3), Thing()
        k.put(("a", 1), (frames, vae), torch.rand(2, 3))
        assert k.get(("a", 1), (frames.clone(), vae)) is None
        assert k.get(("a", 1), (frames, Thing())) is None
    case("a new tensor of equal content, or another VAE, misses", equal_content_misses)

    def changed_value_misses():
        k, frames = keep(), torch.rand(4, 8, 8, 3)
        k.put(("a", 1), (frames,), torch.rand(2, 3))
        assert k.get(("a", 2), (frames,)) is None
    case("a changed plain value misses", changed_value_misses)

    def dead_object_empties():
        k, frames, vae = keep(), torch.rand(4, 8, 8, 3), Thing()
        k.put(("a", 1), (frames, vae), torch.rand(2, 3))
        k.put(("a", 2), (frames, vae), torch.rand(2, 3))
        assert len(k.entries) == 2
        del frames
        gc.collect()
        assert len(k.entries) == 0, f"{len(k.entries)} entries outlived the frames they were made from"
    case("an entry goes when the frames it was made from are collected", dead_object_empties)

    def store_does_not_pin():
        import weakref
        k, frames = keep(), torch.rand(4, 8, 8, 3)
        seen = weakref.ref(frames)
        k.put(("a", 1), (frames,), torch.rand(2, 3))
        del frames
        gc.collect()
        assert seen() is None, "the store kept the loader's frames alive"
    case("the store does not keep the loaded frames alive", store_does_not_pin)

    # --------------------------------------------------------------- budget
    def oldest_goes_first():
        k, obj = keep(budget=2 * MiB + 1024), Thing()
        one = lambda: torch.zeros(MiB // 4, dtype=torch.float32)      # a MiB each
        for name in "abc":
            k.put((name,), (obj,), one())
            if name == "b":
                assert k.get(("a",), (obj,)) is not None                # `a` is now the more recent
        names = [key[0][0] for key in k.entries]
        assert names == ["a", "c"], names
        return str(names)
    case("past the budget the least recently used entry goes", oldest_goes_first)

    def roomy_keeps_all():
        k, obj = keep(budget=8 * MiB), Thing()
        for name in "abc":
            k.put((name,), (obj,), torch.zeros(MiB // 4, dtype=torch.float32))
        assert len(k.entries) == 3
    case("RED CONTROL: with room for all three nothing is evicted", roomy_keeps_all)

    def too_large_alone():
        k, obj = keep(budget=MiB // 2), Thing()
        assert k.put(("a",), (obj,), torch.zeros(MiB // 4, dtype=torch.float32)) is False
        assert len(k.entries) == 0
    case("a value over the budget alone is not kept", too_large_alone)

    def short_host_keeps_nothing():
        k, obj = keep(short=lambda: True), Thing()
        assert k.put(("a",), (obj,), torch.rand(2, 3)) is False
        assert k.get(("a",), (obj,)) is None
    case("a host short of memory keeps nothing", short_host_keeps_nothing)

    # ------------------------------------------------------------- on the host
    def moved_to_host():
        # A meta tensor is on a device that is not the CPU and needs no card. It has no data
        # to copy, so the mover is stood in for; what is held is that every such tensor goes
        # through it, at any depth, and the CPU ones are left as they are.
        cpu, elsewhere = torch.rand(2, 3), torch.empty(2, 3, device="meta")
        moved = []

        def move(t):
            moved.append(t)
            return torch.zeros(tuple(t.shape))
        out = wk._to_host([[cpu, {"refs": [{"latent": elsewhere}], "n": 3}]], move=move)
        assert len(moved) == 1 and moved[0] is elsewhere, "the tensor off the CPU was not moved, or one on it was"
        assert out[0][0] is cpu, "a tensor already on the CPU was copied"
        assert all(t.device.type == "cpu" for t in wk._tensors(out))
        assert out[0][1]["n"] == 3
    case("every tensor off the CPU goes through the one mover, at any depth", moved_to_host)

    def put_uses_it():
        k, obj = keep(), Thing()
        k.put(("a",), (obj,), [[torch.rand(2, 3), {"x": torch.rand(4)}]])
        (entry,) = k.entries.values()
        assert all(t.device.type == "cpu" for t in wk._tensors(entry.value))
    case("what `put` holds is on the CPU", put_uses_it)

    # ---------------------------------------------------------- not written to
    def cond():
        return [[torch.rand(1, 16, 8), {"pooled_output": None, "minimax_refs": [{"latent": torch.rand(1, 4, 2, 2)}]}]]

    def untouched_hits():
        k, obj = keep(), Thing()
        k.put(("c",), (obj,), cond())
        assert k.get(("c",), (obj,)) is not None and k.get(("c",), (obj,)) is not None
        assert k.written_to == 0
    case("RED CONTROL: a kept conditioning nobody wrote to hits, twice", untouched_hits)

    def written_to_is_dropped():
        for where in ("the rows", "a reference latent"):
            k, obj = keep(), Thing()
            k.put(("c",), (obj,), cond())
            got = k.get(("c",), (obj,))
            target = got[0][0] if where == "the rows" else got[0][1]["minimax_refs"][0]["latent"]
            target[..., 0] += 1.0                                       # a consumer writing in place
            assert k.get(("c",), (obj,)) is None, f"a write into {where} still hit"
            assert k.written_to == 1 and len(k.entries) == 0
    case("a kept conditioning written into is dropped at the next get, and counted", written_to_is_dropped)

    def containers_are_fresh():
        k, obj = keep(), Thing()
        k.put(("c",), (obj,), cond())
        got = k.get(("c",), (obj,))
        got.append("extra")
        got[0][1]["added"] = 1
        again = k.get(("c",), (obj,))
        assert len(again) == 1 and "added" not in again[0][1]
        assert again[0][0] is got[0][0], "the tensors themselves are shared, not copied"
    case("what get returns can be appended to without changing what is kept", containers_are_fresh)

    def bf16_digests():
        a = torch.rand(3, 5).to(torch.bfloat16)
        b = a.clone()
        assert wk.digest([a]) == wk.digest([b])
        b[0, 0] += 1
        assert wk.digest([a]) != wk.digest([b])
        assert wk.digest([torch.zeros(0, 4)]) != wk.digest([torch.zeros(0, 5)])
    case("the digest reads a bfloat16 tensor and an empty one", bf16_digests)

    # -------------------------------------------------------------- latent key
    class Vae:
        def __init__(self, dtype=torch.float16, held=torch.float16):
            self.vae_dtype = dtype
            self.first_stage_model = types.SimpleNamespace(encoder=torch.nn.Linear(2, 2).to(held))

    def source(**over):
        base = {"frames": torch.rand(6, 8, 8, 3), "mask": torch.zeros(6, 8, 8), "grow_pixels": 64,
                "paint_out": False, "start_from": wk.START_NOISE, "start_blur": 16,
                "motion_reference": wk.MOTION_NONE, "motion_short_edge": 384, "motion_vae": False}
        base.update(over)
        return base

    def hits(key_a, key_b) -> bool:
        k = keep()
        k.put(*key_a, torch.rand(2, 3))
        return k.get(*key_b) is not None

    def latent_mask_only_when_encoded():
        vae, src = Vae(), source()
        other_mask = dict(src, mask=torch.zeros(6, 8, 8))
        assert hits(wk.latent_key(src, vae, 0, 5, 64, 32), wk.latent_key(other_mask, vae, 0, 5, 64, 32)), \
            "a new mask on an unchanged plate should hit"
        for change in ({"paint_out": True}, {"start_from": "top"}):
            a, b = dict(src, **change), dict(other_mask, **change)
            assert hits(wk.latent_key(a, vae, 0, 5, 64, 32), wk.latent_key(a, vae, 0, 5, 64, 32))
            assert not hits(wk.latent_key(a, vae, 0, 5, 64, 32), wk.latent_key(b, vae, 0, 5, 64, 32)), \
                f"with {change} the mask changes what is encoded and a new mask still hit"
            assert not hits(wk.latent_key(src, vae, 0, 5, 64, 32), wk.latent_key(a, vae, 0, 5, 64, 32))
    case("the latent key holds the mask only under paint_out or a late start", latent_mask_only_when_encoded)

    def latent_margin_rule():
        vae = Vae()
        fixed = source(paint_out=True, grow_by=wk.GROW_FIXED, feather_pixels=2)
        scaled = dict(fixed, grow_by=wk.GROW_SUBJECT)
        key = lambda s: wk.latent_key(s, vae, 0, 5, 64, 32)
        assert not hits(key(fixed), key(scaled)), "an encode painted out under one margin rule was handed to the other"
        assert not hits(key(scaled), key(dict(scaled, feather_pixels=12))), \
            "under the subject's size the feather is the margin's floor, and another feather still hit"
        assert hits(key(fixed), key(dict(fixed, feather_pixels=12))), "under a fixed margin the feather does not move the hole"
        plain = source(grow_by=wk.GROW_FIXED, feather_pixels=2)
        assert hits(key(plain), key(dict(plain, grow_by=wk.GROW_SUBJECT))), \
            "with nothing painted out the plate's encode does not depend on the margin"
    case("the latent key holds the margin's rule, and its floor when the rule reads the subject", latent_margin_rule)

    def latent_window_canvas_vae():
        vae, src = Vae(), source()
        base = wk.latent_key(src, vae, 0, 5, 64, 32)
        assert hits(base, wk.latent_key(src, vae, 0, 5, 64, 32))
        for other in (wk.latent_key(src, vae, 5, 5, 64, 32), wk.latent_key(src, vae, 0, 3, 64, 32),
                      wk.latent_key(src, vae, 0, 5, 32, 32), wk.latent_key(src, Vae(), 0, 5, 64, 32)):
            assert not hits(base, other)
        # the precision node casts the loaded module in place: same object, another dtype
        before = wk.latent_key(src, vae, 0, 5, 64, 32)
        vae.vae_dtype = torch.float32
        vae.first_stage_model.encoder.to(torch.float32)
        assert not hits(before, wk.latent_key(src, vae, 0, 5, 64, 32)), "an encoder cast in place still hit"
        late = dict(src, start_from="top")
        assert not hits(wk.latent_key(late, vae, 0, 5, 64, 32), wk.latent_key(dict(late, start_blur=8), vae, 0, 5, 64, 32))
    case("the latent key moves with the window, the canvas, the VAE and its dtypes", latent_window_canvas_vae)

    # ---------------------------------------------------------------- cond key
    class Clip:
        def __init__(self):
            self.patcher = types.SimpleNamespace(patches_uuid=uuid.uuid4())
            self.layer_idx = None

    def cond_key_parts():
        clip, vae, audio_vae, still = Clip(), Vae(), Thing(), Thing()
        src = source()
        args = (clip, "a prompt", 345, 64, 32, (still,), vae, audio_vae, src, 0)
        base = wk.cond_key(*args)
        assert hits(base, wk.cond_key(*args))
        new_mask = dict(src, mask=torch.zeros(6, 8, 8), frames=torch.rand(6, 8, 8, 3))
        assert hits(base, wk.cond_key(clip, "a prompt", 345, 64, 32, (still,), vae, audio_vae, new_mask, 24)), \
            "with no motion reference the frames, the mask and the window's start are not in the conditioning"
        assert hits(base, wk.cond_key(clip, "a prompt", 345, 64, 32, [still], vae, audio_vae, None, 0)), \
            "the same record in a list, with no source at all, is the same conditioning"
        for other in (wk.cond_key(clip, "another prompt", 345, 64, 32, (still,), vae, audio_vae, src, 0),
                      wk.cond_key(clip, "a prompt", 141, 64, 32, (still,), vae, audio_vae, src, 0),
                      wk.cond_key(clip, "a prompt", 345, 64, 32, (Thing(),), vae, audio_vae, src, 0),
                      wk.cond_key(clip, "a prompt", 345, 64, 32, (still, Thing()), vae, audio_vae, src, 0),
                      wk.cond_key(Clip(), "a prompt", 345, 64, 32, (still,), vae, audio_vae, src, 0),
                      wk.cond_key(clip, "a prompt", 345, 64, 32, (still,), vae, None, src, 0)):
            assert not hits(base, other)
        before = wk.cond_key(*args)
        clip.patcher.patches_uuid = uuid.uuid4()                        # a LoRA patched onto the encoder
        assert not hits(before, wk.cond_key(*args)), "new patches on the encoder still hit"
    case("the conditioning key moves with the text, the length, each reference and the encoder's patches", cond_key_parts)

    def cond_key_motion():
        clip, vae, still = Clip(), Vae(), Thing()
        src = source(motion_reference="subject only")
        args = lambda s, first=0: (clip, "a prompt", 345, 64, 32, (still,), vae, None, s, first)
        base = wk.cond_key(*args(src))
        assert hits(base, wk.cond_key(*args(src)))
        for other in (dict(src, mask=torch.zeros(6, 8, 8)), dict(src, frames=torch.rand(6, 8, 8, 3)),
                      dict(src, motion_short_edge=256), dict(src, grow_pixels=32), dict(src, motion_vae=True),
                      dict(src, motion_reference="whole frame")):
            assert not hits(base, wk.cond_key(*args(other)))
        assert not hits(base, wk.cond_key(*args(src, first=306))), "another window's frames still hit"
        assert not hits(wk.cond_key(*args(source())), base), "a motion reference turned on still hit"
    case("with a motion reference the conditioning key holds the window's frames, mask and settings", cond_key_motion)

    def cond_key_zoom():
        clip, vae, still = Clip(), Vae(), Thing()
        boxes = torch.tensor([[2, 1, 6, 7]] * 6)
        table = '{"shots": [{"first_frame": 0, "last_frame": 5}]}'
        src = source(motion_reference=wk.MOTION_ZOOM, subject_boxes=boxes, shot_table=table)
        args = lambda s: (clip, "a prompt", 6, 64, 32, (still,), vae, None, s, 0)
        base = wk.cond_key(*args(src))
        assert hits(base, wk.cond_key(*args(dict(src, subject_boxes=boxes.clone())))), "the same boxes in another tensor missed"
        moved = boxes.clone(); moved[3, 2] = 7
        assert not hits(base, wk.cond_key(*args(dict(src, subject_boxes=moved)))), "a moved box still hit"
        cut = '{"shots": [{"first_frame": 0, "last_frame": 2}, {"first_frame": 3, "last_frame": 5}]}'
        assert not hits(base, wk.cond_key(*args(dict(src, shot_table=cut)))), "another cut still hit"
        assert not hits(base, wk.cond_key(*args(dict(src, motion_reference="subject only")))), "the zoom turned off still hit"
        plain = source(motion_reference="subject only")
        assert hits(wk.cond_key(*args(plain)), wk.cond_key(*args(dict(plain, subject_boxes=moved)))), \
            "boxes moved a key that does not read them"
    case("zoomed in, the conditioning key holds the window's boxes and the shot table: a changed box misses, an unchanged one hits", cond_key_zoom)

    def cond_key_wired():
        clip, vae, still = Clip(), Vae(), Thing()
        boxes = torch.tensor([[2, 1, 6, 7]] * 6)
        table = '{"shots": [{"first_frame": 0, "last_frame": 5}]}'
        video = torch.rand(6, 8, 8, 3)
        args = lambda s: (clip, "a prompt", 6, 64, 32, (still,), vae, None, s, 0)
        for mode in wk.WIRED:
            src = source(motion_reference=mode, subject_boxes=boxes, shot_table=table, motion_frames=video)
            base = wk.cond_key(*args(src))
            assert hits(base, wk.cond_key(*args(dict(src)))), f"{mode}: the same wired video missed"
            assert not hits(base, wk.cond_key(*args(dict(src, motion_frames=torch.rand(6, 8, 8, 3))))), \
                f"{mode}: another wired video on the same source still hit"
            moved = boxes.clone(); moved[3, 2] = 7
            reads_boxes = mode in wk.ZOOMED
            assert hits(base, wk.cond_key(*args(dict(src, subject_boxes=moved)))) != reads_boxes, \
                f"{mode}: a moved box {'still hit' if reads_boxes else 'missed, though this choice does not read boxes'}"
        one, two = (wk.cond_key(*args(source(motion_reference=m, subject_boxes=boxes, shot_table=table, motion_frames=video)))
                    for m in wk.WIRED)
        assert not hits(one, two), "the wired video whole and zoomed share a key"
    case("a wired motion video is in the conditioning key itself, and zoomed in so are the boxes and the shot table", cond_key_wired)

    def cond_key_no_source():
        # a song graph with no Masked Source, and one with no references either
        clip, vae = Clip(), Vae()
        plain = wk.cond_key(clip, "a prompt", 345, 64, 32, None, vae, None, None, 0)
        assert hits(plain, wk.cond_key(clip, "a prompt", 345, 64, 32, None, vae, None, None, 306)), \
            "with no source the window's start is not in the conditioning"
        assert not hits(plain, wk.cond_key(clip, "another prompt", 345, 64, 32, None, vae, None, None, 0))
        assert not hits(plain, wk.cond_key(clip, "a prompt", 345, 64, 32, (Thing(),), vae, None, None, 0))
        assert not hits(plain, wk.cond_key(Clip(), "a prompt", 345, 64, 32, None, vae, None, None, 0))
    case("the conditioning key builds with no source and no references, for a song graph that has neither", cond_key_no_source)

    # ------------------------------------------------------ restated constants
    def constants():
        vm = load("video_mask")
        assert wk.START_NOISE == vm.START_NOISE and wk.MOTION_NONE == vm.MOTION_NONE and wk.MOTION_ZOOM == vm.MOTION_ZOOM
        assert wk.ZOOMED == vm.ZOOMED and wk.WIRED == vm.WIRED, (wk.ZOOMED, vm.ZOOMED, wk.WIRED, vm.WIRED)
        assert wk.GROW_FIXED == vm.GROW_FIXED and wk.GROW_SUBJECT == vm.GROW_SUBJECT
        return f"{wk.START_NOISE!r}, {wk.MOTION_NONE!r}, {wk.MOTION_ZOOM!r}"
    case("START_NOISE, MOTION_NONE and MOTION_ZOOM are video_mask.py's", constants)

    def song_node_reads_under_the_switch():
        # Read from the node's source: its execute needs a sampler and a model to run. Every
        # read of a keep must sit on a line that is conditional on `reuse_windows`.
        lines = (REPO / "audio_freeze_song.py").read_text().splitlines()
        reads = [line.strip() for line in lines if "window_keep." in line and ".get(" in line]
        puts = [line.strip() for line in lines if "window_keep." in line and ".put(" in line]
        assert len(reads) == 2, f"expected one read of each keep, found {reads}"
        for line in reads:
            assert "if reuse_windows else None" in line, f"a read that ignores reuse_windows: {line}"
        assert sorted(line.split(".")[1] for line in puts) == ["CONDS", "LATENTS"], puts
        return "two reads under the switch, two puts"
    case("the song node reads the keeps only under reuse_windows, and stores into both", song_node_reads_under_the_switch)

    def module_keeps():
        assert wk.LATENTS.budget == wk.LATENT_BYTES and wk.CONDS.budget == wk.COND_BYTES
        assert wk.LATENTS is not wk.CONDS
    case("the two keeps are separate and take their budgets from the named constants", module_keeps)

    return finish()


if __name__ == "__main__":
    sys.exit(main())
