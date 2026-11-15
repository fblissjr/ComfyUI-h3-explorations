#!/usr/bin/env python3
"""The SAM 3.1 Corrections node on stand-in modules, through ComfyUI's real ModelPatcher. No SAM weights, no card.

`sam31_corrections.py` attaches three corrections to a loaded SAM 3.1 pair as object patches: the image
range at the trunk's patch embedding, the text encoder's activation, and memory selection on the video tracker. This check builds a stand-in with the same
attribute path and a stand-in text encoder made of ComfyUI's own `CLIPMLP`, wraps each in ComfyUI's `ModelPatcher`,
and asks whether a frame reaches the first layer as the node says it does. Each case is a way a graph could run on a
range nobody meant:

  the_path_is_cores               the attribute path the range patch hangs on is spelled the way ComfyUI's SAM 3
                                  classes spell it today (read from their source), and the activation it replaces is
                                  the one the SAM 3 text encoder's config names. A rename upstream is red here.
  a_frame_arrives_in_range        patched and applied by the patcher, a frame in 0..1 reaches the layer as -1..1,
                                  and one that a resize overshot is exactly -1..1. THE CONTROL: the same frame through
                                  the unpatched stand-in arrives as it was, so the case can see a missing patch.
  applied_and_undone_by_comfyui   `patch_model` installs it and `unpatch_model` puts the stock forward back; a frame
                                  after the undo arrives as it was.
  never_twice                     attaching twice, and patching the model twice, still maps a frame once.
  a_clone_gets_its_own_layer      a clone handed a second instance of the model computes with THAT instance's
                                  weights. THE CONTROL: with the clone callback removed the clone computes with the
                                  first instance's, which is the fault the callback exists for.
  never_corrects_what_is_right    a frame that arrives already in -1..1 is passed through untouched, the patch says
                                  so, and from then on a bright mapped frame is untouched too. KNOWN LIMIT, asserted
                                  so it is never read as covered: before the first such frame, a bright mapped frame
                                  has nothing low enough to give it away and is mapped again.
  the_activation_is_swapped       every MLP that ran the shipped activation runs exact GELU when applied, and the
                                  shipped one again when undone; an encoder ComfyUI already built with exact GELU is
                                  left alone and reported so; one with none of ComfyUI's MLPs is refused.
  the_input_pair_stays_stock      the node's own entry point returns clones and leaves the pair it was handed stock;
                                  wired twice in a row it corrects once; and the stock patcher and the corrected clone,
                                  which share one model, each give the layer their own range when applied in turn, in
                                  both orders. (On the plain patcher, applied by hand: not ComfyUI's loading decision.)
  a_loaded_sibling_changes_nothing  the node run again on the stock pair WHILE its first corrected pair is loaded still
                                  attaches the activation patch (the modules then show the loaded clone's patch, not what
                                  they were built with: found by review on 2026-10-07, after the node was served), the
                                  stock pair still reads stock, and a clone of a patch that has seen mapped frames knows it.
  switches_and_report             each correction alone, neither, and both: `corrections` reads exactly what was
                                  attached, and the report says so in words, naming what is never corrected.
  memory_paths_are_cores          the tracker/detector paths and attributes patched by memory selection,
                                  multimask pointer, and detector presence are spelled the way ComfyUI's SAM 3 classes
                                  spell them today, and `SAM31Tracker.num_maskmem = 7` (six past frames).
  memory_on_and_off               on a stand-in SAM 3.1 tracker: switched on, memory patches are attached, installed
                                  by `patch_model` and gone after `unpatch_model`, and the report says on; switched off,
                                  neither is there and the report says ComfyUI's six frames. THE CONTROL: a stock model
                                  reads off.
  memory_never_twice              attached twice, and wired twice through the node's entry point, one memory patch and one
                                  clone callback; a second attach asking another threshold is refused, not swallowed.
  memory_clone_binds_its_own      a clone handed a second instance of the model has a memory patch bound to THAT
                                  instance's tracker and decoder, with a state of its own. THE CONTROL: with the clone
                                  callback removed the clone's patch is the first instance's.
  memory_scored_per_object        three objects in one SAM 3.1 call (the bench tool's attach; explicit focus), the first
                                  hidden on frames 3-12, the others seen throughout: through the patch ComfyUI applied,
                                  the hidden one keeps only the frames it was seen on while the two seen ones keep every
                                  frame.
  memory_left_off_on_sam3         a SAM 3 (`SAM3Tracker`) stand-in with the switch on carries no memory patch, and the
                                  report says it is not applied there and why.
  memory_survives_an_object_added two objects, a third added after frame 4 as core's `_add_detected_objects` adds a
                                  detection (the state grows, that frame's stored scores keep the old count): no error,
                                  frame 4 scored for two and frame 5 for three, the added object keeping the earlier
                                  frames unfiltered; sixteen objects grown to seventeen scored for sixteen.
  memory_holds_nothing_after_a_run  four tracks in a row: after each, no memory tensor and no decoder output is held,
                                  every kept score is on the CPU, and what is kept is the same size every time.
  memory_pointers_as_meta         what core's method is handed: pointers from every kept frame but the oldest (Meta's
                                  rule), and none when only the frame before is kept.
  memory_multiplex_dms_multi_object  Decoupled Multiplex Memory Selection on multi-object calls (`focus_obj_idx=None`):
                                  visible companions keep recent spatial memory when object 0 is hidden (contrasted
                                  with the `focus_obj_idx=0` control), `MemoryEncodeFilter` suppresses unreliable
                                  slots (`-1024.0` mask / `-10.0` score) for `N_obj >= 2` while exempting conditioning,
                                  newborn, and Call #2 reconditioned slots and leaving `N_obj == 1` untouched,
                                  explicit `one_object_only=True` still falls back to core's memory, and 2-bucket
                                  `17`-object `combined_ptr` keeps bucket 1 slot 0 intact.
  memory_a_later_track_scores     two tracks with no reset between, the second on the same output_dict object from a
                                  later frame: it scores and keeps its own frames.
  multimask_pointer_matches_best_iou  `correct_multimask_ptr`: `SAM31Tracker._forward_propagation` projects the
                                  winning `argmax(iou_obj)` mask token instead of hardcoded branch 0 (both standalone
                                  and alongside `memory_selection`), leaves `last_hs` None, rebinds on `.clone()`, and
                                  is undone by `unpatch_model`.
  detector_presence_gates_scores  `correct_detector_presence`: `SAM3Detector._detect` multiplies `sigmoid(presence)`
                                  into prompt logits in `fp32` (safe in `fp16` with no `Inf`/`NaN`), leaves unprompted
                                  calls at zero, rebinds on `.clone()`, is undone by `unpatch_model`, and is optional
                                  default-off on `MiniMaxH3SAM31Corrections`.

What it cannot check: the server's dynamic patcher class (this builds the plain `ModelPatcher`), the real model, and
that ComfyUI's own SAM nodes reach the patched layer. That is a real-model run as a graph on the server's queue. For
memory selection the trackers are core's own classes built small with random weights, and the decoder's scores are
set by hand: it shows which frames are kept, not that the kept frames track better (that is
`bench/sam_memory_selection.py`, on the real model).

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_sam31_corrections.py
"""
from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(COMFY))

from _lib import case, finish  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True
import comfy.clip_model  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402
from comfy.ldm.sam3.tracker import MultiplexState, SAM31Tracker, SAM3Tracker  # noqa: E402
from comfy.patcher_extension import CallbacksMP  # noqa: E402

import sam31_corrections as L  # noqa: E402

CPU = torch.device("cpu")


class _PatchEmbed(torch.nn.Module):
    """A patch embedding with no bias, as core's; it remembers the range of what it was last handed."""

    def __init__(self, seed: int):
        super().__init__()
        self.proj = torch.nn.Conv2d(3, 4, 2, stride=2, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(self.proj.weight.shape, generator=torch.Generator().manual_seed(seed)))
        self.seen = None

    def forward(self, x):
        self.seen = (float(x.min()), float(x.max()))
        return self.proj(x)


def _model(seed: int = 0) -> torch.nn.Module:
    """A stand-in with `PATCH_EMBED`'s path: diffusion_model.detector.backbone["vision_backbone"].trunk.patch_embed."""
    trunk = torch.nn.Module()
    trunk.patch_embed = _PatchEmbed(seed)
    vision = torch.nn.Module()
    vision.trunk = trunk
    detector = torch.nn.Module()
    detector.backbone = torch.nn.ModuleDict({"vision_backbone": vision})
    sam = torch.nn.Module()
    sam.detector = detector
    base = torch.nn.Module()
    base.diffusion_model = sam
    return base


def _patcher(model) -> comfy.model_patcher.ModelPatcher:
    return comfy.model_patcher.ModelPatcher(model, load_device=CPU, offload_device=CPU)


def _embed(model) -> _PatchEmbed:
    return model.diffusion_model.detector.backbone["vision_backbone"].trunk.patch_embed


def _clip(shipped: bool = True):
    """A stand-in CLIP: two of core's own MLPs under a patcher, with the attributes `sam31_corrections` reads."""
    enc = torch.nn.Module()
    enc.layers = torch.nn.ModuleList()
    for _ in range(2):
        layer = torch.nn.Module()
        layer.mlp = comfy.clip_model.CLIPMLP(8, 16, L.SHIPPED_ACTIVATION if shipped else "gelu", torch.float32, "cpu", comfy.ops.manual_cast)
        with torch.no_grad():      # core's ops leave weights uninitialised; the case compares outputs, so give them values
            for k, t in enumerate(layer.mlp.parameters()):
                t.copy_(torch.randn(t.shape, generator=torch.Generator().manual_seed(11 + k)))
        enc.layers.append(layer)
    return types.SimpleNamespace(patcher=_patcher(enc), cond_stage_model=enc)


def _two(model, clip) -> tuple[bool, int]:
    got = L.corrections(model, clip)
    return got["image_range"], got["text_activation_layers"]


def _frame(low: float = 0.0, high: float = 1.0) -> torch.Tensor:
    x = torch.rand((1, 3, 8, 8), generator=torch.Generator().manual_seed(3))
    x = (x - x.min()) / (x.max() - x.min())
    return x * (high - low) + low


def the_path_is_cores():
    import comfy.ldm.sam3.detector as detector
    import comfy.ldm.sam3.sam as sam
    import comfy.text_encoders.sam3_clip as sam3_clip
    want = {"diffusion_model": None, "detector": inspect.getsource(detector.SAM3Model), "backbone": inspect.getsource(detector.SAM3Detector),
            "vision_backbone": inspect.getsource(detector.SAM3Detector), "trunk": inspect.getsource(sam.SAM3VisionBackbone),
            "patch_embed": inspect.getsource(sam.ViTDet)}
    assert L.PATCH_EMBED.split(".") == list(want), f"PATCH_EMBED is {L.PATCH_EMBED}"
    for name, source in want.items():
        if source is not None:
            assert f"self.{name} =" in source or f'"{name}"' in source, f"core's SAM 3 classes no longer spell `{name}`"
    assert f'"hidden_act": "{L.SHIPPED_ACTIVATION}"' in inspect.getsource(sam3_clip), "core's SAM 3 text config names another activation"
    assert L.SHIPPED_ACTIVATION in comfy.clip_model.ACTIVATIONS
    return "five names read from core's source; the text config names " + L.SHIPPED_ACTIVATION


def a_frame_arrives_in_range():
    stock = _model()
    _embed(stock)(_frame())
    assert _embed(stock).seen == (0.0, 1.0), f"the control: an unpatched layer saw {_embed(stock).seen}"
    model = _model()
    p = _patcher(model)
    L.attach(p, None, text_activation=False)
    p.patch_model(load_weights=False)
    try:
        _embed(model)(_frame())
        assert _embed(model).seen == (-1.0, 1.0), f"0..1 arrived as {_embed(model).seen}"
        _embed(model)(_frame(-0.07, 1.08))
        assert _embed(model).seen == (-1.0, 1.0), f"an overshot frame arrived as {_embed(model).seen}"
        out = _embed(model)(_frame())
        assert torch.equal(out, _embed(model).proj(_frame() * 2 - 1)), "the layer's output is not its stock output on the mapped frame"
    finally:
        p.unpatch_model(unpatch_weights=False)
    return "0..1 and an overshot frame both arrive as exactly -1..1; unpatched, 0..1"


def applied_and_undone_by_comfyui():
    model = _model()
    p = _patcher(model)
    L.attach(p, None, text_activation=False)
    assert "forward" not in vars(_embed(model)), "attached before ComfyUI applied it"
    p.patch_model(load_weights=False)
    assert isinstance(vars(_embed(model)).get("forward"), L.CorrectedRange), "patch_model did not install the patch"
    p.unpatch_model(unpatch_weights=False)
    assert not isinstance(_embed(model).forward, L.CorrectedRange), "unpatch_model left the patch on"
    _embed(model)(_frame())
    assert _embed(model).seen == (0.0, 1.0), f"after the undo a frame arrived as {_embed(model).seen}"
    return "installed by patch_model, gone after unpatch_model"


def never_twice():
    model = _model()
    p = _patcher(model)
    L.attach(p, None, text_activation=False)
    first = p.object_patches[L.RANGE_PATCH]
    L.attach(p, None, text_activation=False)
    assert p.object_patches[L.RANGE_PATCH] is first, "a second attach replaced the patch"
    assert len(p.get_all_callbacks(CallbacksMP.ON_CLONE)) == 1, "a second attach added a second clone callback"
    p.patch_model(load_weights=False)
    p.patch_model(load_weights=False)
    try:
        L.attach(p.clone(), None, text_activation=False)      # attaching to a clone while the model is patched
        _embed(model)(_frame(0.25, 0.75))
        assert _embed(model).seen == (-0.5, 0.5), f"a frame in 0.25..0.75 arrived as {_embed(model).seen}; mapped twice would be -2..0"
    finally:
        p.unpatch_model(unpatch_weights=False)
    return "one patch, one callback, one mapping"


def a_clone_gets_its_own_layer():
    def run(with_callback: bool) -> tuple[bool, bool]:
        first, second = _model(seed=1), _model(seed=2)
        p = _patcher(first)
        L.attach(p, None, text_activation=False)
        if not with_callback:
            p.remove_callbacks_with_key(CallbacksMP.ON_CLONE, "h3_sam31_range")
        clone = p.clone(model_override=(second, ({}, {}, {}, set())))
        clone.patch_model(load_weights=False)
        try:
            out = _embed(second)(_frame())
        finally:
            clone.unpatch_model(unpatch_weights=False)
        mapped = _frame() * 2 - 1
        return torch.equal(out, _embed(second).proj(mapped)), torch.equal(out, _embed(first).proj(mapped))

    own, others = run(True)
    assert own and not others, "a clone with a second model instance did not compute with its own layer"
    own, others = run(False)
    assert others and not own, "the control did not show the fault: without the callback the clone should use the first instance's layer"
    return "its own weights with the callback; the first instance's without it"


def never_corrects_what_is_right():
    model = _model()
    p = _patcher(model)
    L.attach(p, None, text_activation=False)
    p.patch_model(load_weights=False)
    try:
        bright = _frame(0.7, 1.0) * 2 - 1                    # mapped, and nothing in it is below 0.4
        assert float(bright.min()) > L.ALREADY_MAPPED_BELOW
        _embed(model)(bright)                                 # the known limit: mapped again, before any frame gives it away
        assert _embed(model).seen == (-0.2, 1.0) or abs((_embed(model).seen or (0, 0))[0] + 0.2) < 1e-6, f"the bright mapped frame arrived as {_embed(model).seen}"
        assert not L.corrections(p, None)["frames_arrive_mapped"]
        _embed(model)(_frame() * 2 - 1)
        assert _embed(model).seen == (-1.0, 1.0), f"a frame already in -1..1 arrived as {_embed(model).seen}"
        assert L.corrections(p, None)["frames_arrive_mapped"], "the patch did not say frames arrive mapped"
        _embed(model)(bright)
        low, high = _embed(model).seen or (0.0, 0.0)
        assert abs(low - 0.4) < 1e-6 and high == 1.0, f"after that, a bright mapped frame was still mapped: {low}..{high}"
        assert "already in -1..1" in L.report(p, None)
    finally:
        p.unpatch_model(unpatch_weights=False)
    return "untouched once a frame shows the range is already right; KNOWN LIMIT: a bright mapped frame before that is mapped again"


def the_activation_is_swapped():
    clip = _clip()
    mlps = [layer.mlp for layer in clip.cond_stage_model.layers]
    shipped = comfy.clip_model.ACTIVATIONS[L.SHIPPED_ACTIVATION]
    L.attach(_patcher(_model()), clip, image_range=False)
    assert all(m.activation is shipped for m in mlps), "attached before ComfyUI applied it"
    clip.patcher.patch_model(load_weights=False)
    assert all(m.activation is F.gelu for m in mlps), "an MLP still runs the shipped activation"
    x = torch.randn((1, 3, 8), generator=torch.Generator().manual_seed(5))
    exact = mlps[0](x)
    clip.patcher.unpatch_model(unpatch_weights=False)
    assert all(m.activation is shipped for m in mlps), "the shipped activation was not put back"
    assert not torch.equal(exact, mlps[0](x)), "the two activations gave the same output, so the case sees nothing"
    exact = _clip(shipped=False)                             # as if ComfyUI built the encoder with exact GELU already
    L.attach(_patcher(_model()), exact, image_range=False)
    got = L.corrections(None, exact)
    assert got["text_activation_layers"] == 0 and got["text_activation_already_exact"] == 2, f"an already exact encoder reads as {got}"
    assert "already exact" in L.report(None, exact)
    empty = types.SimpleNamespace(patcher=_patcher(torch.nn.Linear(2, 2)))
    try:
        L.attach(_patcher(_model()), empty, image_range=False)
    except L.NotSAM3:
        pass
    else:
        raise AssertionError("an encoder with none of ComfyUI's MLPs was not refused")
    return f"{len(mlps)} MLPs: exact GELU applied, the shipped one back after the undo; an already exact encoder left alone"


def switches_and_report():
    seen = {}
    for image_range in (True, False):
        for activation in (True, False):
            p, clip = _patcher(_model()), _clip()
            L.attach(p, clip, image_range=image_range, text_activation=activation)
            got = _two(p, clip)
            assert got == (image_range, 2 if activation else 0), f"{image_range}, {activation}: {got}"
            text = L.report(p, clip)
            assert ("image range: corrected" in text) == image_range and ("image range: NOT corrected" in text) == (not image_range), text
            assert ("activation: corrected" in text) == activation and ("activation: NOT corrected" in text) == (not activation), text
            assert all(item in text for item in L.NOT_CORRECTED), "the report does not name what is never corrected"
            seen[(image_range, activation)] = text
    assert len(set(seen.values())) == 4
    stock = L.corrections(_patcher(_model()), _clip())
    assert not any(stock.values()), f"a stock pair reads as {stock}"
    return "four combinations read back as attached; a stock pair reads as uncorrected"


def the_input_pair_stays_stock():
    model, enc = _model(), _clip()
    p = _patcher(model)
    stand_in_clip = types.SimpleNamespace(patcher=enc.patcher, cond_stage_model=enc.cond_stage_model,
                                          clone=lambda: types.SimpleNamespace(patcher=enc.patcher.clone(), cond_stage_model=enc.cond_stage_model))
    m2, c2 = L.corrected(p, stand_in_clip)
    assert m2 is not p and c2.patcher is not enc.patcher, "the node returned the pair it was handed"
    assert _two(p, stand_in_clip) == (False, 0), "the pair handed in was patched"
    assert _two(m2, c2) == (True, 2)
    c2.clone = lambda: types.SimpleNamespace(patcher=c2.patcher.clone(), cond_stage_model=c2.cond_stage_model)
    m3, c3 = L.corrected(m2, c2)                               # wired twice in a row
    assert _two(m3, c3) == (True, 2)
    assert len(m3.get_all_callbacks(CallbacksMP.ON_CLONE)) == 1, "wired twice, the clone callback is there twice"
    # the stock patcher and the corrected clone share ONE model: each applied in turn, the layer sees what that one says
    seen = []
    for which in (p, m3, p, m3, m3, p):
        which.patch_model(load_weights=False)
        try:
            _embed(model)(_frame(0.25, 0.75))
            seen.append(_embed(model).seen)
        finally:
            which.unpatch_model(unpatch_weights=False)
    assert seen == [(0.25, 0.75), (-0.5, 0.5), (0.25, 0.75), (-0.5, 0.5), (-0.5, 0.5), (0.25, 0.75)], f"stock and corrected in turn saw {seen}"
    return "the pair handed in reads stock; twice in a row is one correction; stock and corrected in turn each see their own range"


def a_loaded_sibling_changes_nothing():
    """The encoder and the model are shared by every clone, and a loaded clone's patches are what the modules show."""
    clip = _clip()
    model = _model()
    p = _patcher(model)
    stand_in = types.SimpleNamespace(patcher=clip.patcher, cond_stage_model=clip.cond_stage_model,
                                     clone=lambda: types.SimpleNamespace(patcher=clip.patcher.clone(), cond_stage_model=clip.cond_stage_model))
    m1, c1 = L.corrected(p, stand_in)
    c1.patcher.patch_model(load_weights=False)             # the first corrected pair is LOADED when the node runs again
    m1.patch_model(load_weights=False)
    try:
        assert all(layer.mlp.activation is F.gelu for layer in clip.cond_stage_model.layers), "the stand-in's patch is not applied"
        m2, c2 = L.corrected(p, stand_in)                  # ... on the stock pair, as a re-run of the node would
        got = L.corrections(m2, c2)
        assert got["text_activation_layers"] == 2 and got["text_activation_already_exact"] == 0, \
            f"with a corrected sibling loaded, a new corrected pair reads as {got}: the live activation was mistaken for the build's"
        assert got["image_range"], got
        assert not any(L.corrections(p, stand_in).values()), "the stock pair reads as corrected while a sibling is loaded"
        _embed(model)(_frame() * 2 - 1)                    # the loaded clone sees a frame that is already mapped ...
        assert L.corrections(m1, c1)["frames_arrive_mapped"]
        assert L.corrections(m1.clone(), c1)["frames_arrive_mapped"], "a clone forgot that frames arrive mapped"
    finally:
        m1.unpatch_model(unpatch_weights=False)
        c1.patcher.unpatch_model(unpatch_weights=False)
    return "a second run of the node while the first pair is loaded still corrects; what the patch has seen is shared by its clones"


def _tracked(kind=SAM31Tracker, seed: int = 0) -> torch.nn.Module:
    """`_model()` with one of core's own trackers at `diffusion_model.tracker`, built small with random weights."""
    torch.manual_seed(seed)
    if kind is SAM31Tracker:
        tracker = SAM31Tracker(d_model=32, mem_dim=32, num_maskmem=4, num_multiplex=16, device=CPU, dtype=torch.float32,
                               operations=comfy.ops.disable_weight_init)
    else:
        tracker = SAM3Tracker(d_model=32, mem_dim=8, num_maskmem=4, device=CPU, dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    tracker.image_size, tracker.backbone_stride = 112, 14
    with torch.no_grad():
        for k, t in enumerate(tracker.parameters()):
            t.copy_(torch.randn(t.shape, generator=torch.Generator().manual_seed(200 + seed + k)) * 0.05)
    tracker.eval()
    model = _model(seed)
    model.diffusion_model.tracker = tracker
    return model


def _memory(p):
    return p.object_patches.get(L.MEMORY_PATCH), p.object_patches.get(L.DECODER_PATCH)


def memory_paths_are_cores():
    import comfy.ldm.sam3.detector as detector
    import comfy.ldm.sam3.tracker as tracker
    assert L.TRACKER.split(".") == ["diffusion_model", "tracker"], f"TRACKER is {L.TRACKER}"
    assert L.DETECTOR.split(".") == ["diffusion_model", "detector"], f"DETECTOR is {L.DETECTOR}"
    assert "self.tracker =" in inspect.getsource(detector), "core's SAM 3 model no longer builds `self.tracker`"
    assert "self.detector =" in inspect.getsource(detector.SAM3Model), "core's SAM 3 model no longer builds `self.detector`"
    for cls in (tracker.SAM3Tracker, tracker.SAM31Tracker):
        assert "_prepare_memory_conditioned_features" in vars(cls), f"{cls.__name__} has no `_prepare_memory_conditioned_features`"
        assert "_encode_new_memory" in vars(cls), f"{cls.__name__} has no `_encode_new_memory`"
        assert "self.sam_mask_decoder =" in inspect.getsource(cls), f"{cls.__name__} no longer builds `self.sam_mask_decoder`"
    assert "_forward_propagation" in vars(tracker.SAM31Tracker), "SAM31Tracker has no `_forward_propagation`"
    assert "track_video_with_detection" in vars(tracker.SAM31Tracker), "SAM31Tracker has no `track_video_with_detection`"
    assert hasattr(tracker.MultiplexMaskDecoder, "transformer") or "self.transformer =" in inspect.getsource(tracker.MultiplexMaskDecoder)
    assert "_detect" in vars(detector.SAM3Detector), "SAM3Detector has no `_detect`"
    assert L.MEMORY_PATCH == L.TRACKER + "._prepare_memory_conditioned_features"
    assert L.DECODER_PATCH == L.TRACKER + ".sam_mask_decoder.forward"
    assert L.ENCODE_MEM_PATCH == L.TRACKER + "._encode_new_memory"
    assert L.TRACK_LOOP_PATCH == L.TRACKER + ".track_video_with_detection"
    assert L.MULTIMASK_TAP_PATCH == L.TRACKER + ".sam_mask_decoder.transformer.forward"
    assert L.PROPAGATION_PATCH == L.TRACKER + "._forward_propagation"
    assert L.PRESENCE_PATCH == L.DETECTOR + "._detect"
    assert L.NO_OBJ_SCORE == tracker.NO_OBJ_SCORE
    assert L.MEMORY_SELECTION_TRACKERS == (tracker.SAM31Tracker.__name__,)
    assert L.MULTIMASK_PTR_TRACKERS == (tracker.SAM31Tracker.__name__,)
    kept = inspect.signature(tracker.SAM31Tracker.__init__).parameters["num_maskmem"].default
    assert kept == 7, f"SAM 3.1's tracker remembers num_maskmem={kept}: the report and the tooltip say six past frames"
    return "tracker, decoder, encode, propagation and detector paths read from core's source; SAM31Tracker num_maskmem=7"


def memory_on_and_off():
    stock = _patcher(_tracked())
    assert _memory(stock) == (None, None) and not any(L.corrections(stock, _clip()).values()), "the control: a stock model reads corrected"
    assert "memory selection: off" in L.report(stock, None), L.report(stock, None)
    model = _tracked()
    tracker = model.diffusion_model.tracker
    on = _patcher(model)
    L.attach(on, None, image_range=False, text_activation=False)            # the switch's default is on
    step, tap = _memory(on)
    enc = on.object_patches.get(L.ENCODE_MEM_PATCH)
    loop = on.object_patches.get(L.TRACK_LOOP_PATCH)
    assert isinstance(step, L.MemorySelectionStep) and isinstance(tap, L.IoUTap), f"switched on, the patches are {step!r}, {tap!r}"
    assert isinstance(enc, L.MemoryEncodeFilter) and isinstance(loop, L.MemoryTrackLoopGuard), (enc, loop)
    assert step.tracker is tracker and step.iou_tap is tap and step.mf_threshold == L.MF_THRESHOLD
    assert step.encode_filter is enc and enc.iou_tap is tap and step.focus_obj_idx is None and not step.one_object_only
    got = L.corrections(on, None)
    assert got["memory_selection"] and not got["multimask_ptr"] and not got["detector_presence"], got
    assert L.tracker_kind(on) == "SAM31Tracker", got
    assert "memory selection: on" in L.report(on, None), L.report(on, None)
    assert "_prepare_memory_conditioned_features" not in vars(tracker), "attached before ComfyUI applied it"
    on.patch_model(load_weights=False)
    try:
        assert vars(tracker).get("_prepare_memory_conditioned_features") is step, "patch_model did not install the memory patch"
        assert vars(tracker).get("_encode_new_memory") is enc, "patch_model did not install the encode filter"
        assert vars(tracker).get("track_video_with_detection") is loop, "patch_model did not install the loop guard"
        assert vars(tracker.sam_mask_decoder).get("forward") is tap, "patch_model did not install the decoder tap"
    finally:
        on.unpatch_model(unpatch_weights=False)
    assert not isinstance(tracker._prepare_memory_conditioned_features, L.MemorySelectionStep), "unpatch_model left the memory patch on"
    assert not isinstance(tracker._encode_new_memory, L.MemoryEncodeFilter), "unpatch_model left the encode filter on"
    assert not isinstance(tracker.track_video_with_detection, L.MemoryTrackLoopGuard), "unpatch_model left the loop guard on"
    assert not isinstance(tracker.sam_mask_decoder.forward, L.IoUTap), "unpatch_model left the decoder tap on"
    off = _patcher(_tracked())
    L.attach(off, None, image_range=False, text_activation=False, memory_selection=False)
    assert _memory(off) == (None, None), "switched off, a memory patch is attached"
    assert L.ENCODE_MEM_PATCH not in off.object_patches and L.TRACK_LOOP_PATCH not in off.object_patches
    assert not L.corrections(off, None)["memory_selection"]
    text = L.report(off, None)
    assert "memory selection: off" in text and "last six frames" in text, text
    return "on: memory, encode filter, loop guard and decoder tap applied and undone by ComfyUI; off: stock last six frames"


def memory_never_twice():
    p = _patcher(_tracked())
    L.attach(p, None, image_range=False, text_activation=False)
    first = _memory(p)
    L.attach(p, None, image_range=False, text_activation=False)
    assert _memory(p) == first, "a second attach replaced the memory patch"
    assert len(p.callbacks.get(CallbacksMP.ON_CLONE, {}).get(L.MEMORY_CALLBACK, [])) == 1, "a second attach added a second clone callback"
    try:
        L.attach_memory_selection(p, mf_threshold=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("an attach asking another threshold of a model that carries one was not refused")
    assert _memory(p) == first
    clip = _clip()
    stand_in = types.SimpleNamespace(patcher=clip.patcher, cond_stage_model=clip.cond_stage_model,
                                     clone=lambda: types.SimpleNamespace(patcher=clip.patcher.clone(), cond_stage_model=clip.cond_stage_model))
    m2, c2 = L.corrected(p, stand_in)
    c2.clone = lambda: types.SimpleNamespace(patcher=c2.patcher.clone(), cond_stage_model=c2.cond_stage_model)
    m3, _ = L.corrected(m2, c2)                               # wired twice in a row
    assert len(m3.callbacks.get(CallbacksMP.ON_CLONE, {}).get(L.MEMORY_CALLBACK, [])) == 1, "wired twice, the memory callback is there twice"
    step, tap = _memory(m3)
    assert isinstance(step, L.MemorySelectionStep) and step.iou_tap is tap
    assert step.tracker is p.model.diffusion_model.tracker and isinstance(step.orig_fn.__self__, SAM31Tracker)
    assert not isinstance(step.orig_fn, L.MemorySelectionStep) and not isinstance(tap.orig_forward, L.IoUTap), "a patch wraps a patch"
    return "one memory patch and one callback, attached twice or wired twice; another threshold refused"


def memory_clone_binds_its_own():
    def run(with_callback: bool):
        first, second = _tracked(seed=1), _tracked(seed=2)
        p = _patcher(first)
        L.attach(p, None, image_range=False, text_activation=False)
        if not with_callback:
            p.remove_callbacks_with_key(CallbacksMP.ON_CLONE, L.MEMORY_CALLBACK)
        clone = p.clone(model_override=(second, ({}, {}, {}, set())))
        return p, clone, first.diffusion_model.tracker, second.diffusion_model.tracker

    p, clone, first, second = run(True)
    step, tap = _memory(clone)
    assert step.tracker is second and step.orig_fn.__self__ is second, "the clone's memory patch is not bound to its own tracker"
    assert tap.orig_forward.__self__ is second.sam_mask_decoder and step.iou_tap is tap, "the clone's decoder tap is not its own"
    assert _memory(p)[0] is not step and _memory(p)[0].tracker is first, "the parent's patch was changed by the clone"
    step.frame_scores[5] = torch.ones(1)
    assert not _memory(p)[0].frame_scores, "a clone's state is shared with its parent"
    _, clone, first, second = run(False)
    assert _memory(clone)[0].tracker is first, "the control did not show the fault: without the callback the clone keeps the first tracker"
    return "bound to the clone's own tracker and decoder, state of its own; without the callback, the first instance's"


def _group(tracker, n_frames: int, hidden: set[int], n_obj: int = 3, add_at: int | None = None, add_n: int = 0,
           out: dict | None = None, start: int = 1) -> dict:
    """`n_obj` objects in one SAM 3.1 call; object 0 has a negative object score on `hidden`, the others are seen throughout.

    The frames' outputs are written as core's video loop writes them (`track_video_with_detection`); the IoU the decoder
    would give is set on the tap the patch records it on. `add_at`: after that frame's output is written, `add_n` objects
    are added as core's `_add_detected_objects` adds a detection: the multiplex state grows and the frame's stored object
    scores stay at the old count. `out` and `start`: run on an output_dict handed in (the same object), from that frame.
    Returns the output_dict."""
    H = W = 8
    C, M = tracker.d_model, tracker.num_multiplex
    mux = MultiplexState(n_obj, M, CPU, torch.float32)
    torch.manual_seed(99)
    feats, pos = [torch.randn(1, H * W, C)], [torch.zeros(1, H * W, C)]
    out = out if out is not None else {"cond_frame_outputs": {0: {"maskmem_features": torch.randn(1, C, H, W), "maskmem_pos_enc": [torch.zeros(1, C, H, W)],
                                      "image_features": torch.randn(1, H * W, C), "image_pos_enc": torch.zeros(1, H * W, C),
                                      "obj_ptr": torch.zeros(1, M, C), "object_score_logits": torch.full((n_obj, 1), 5.0)}},
           "non_cond_frame_outputs": {}}
    for f in range(start, n_frames):
        tracker._prepare_memory_conditioned_features(frame_idx=f, is_init_cond_frame=False, current_vision_feats=feats,
                                                     current_vision_pos_embeds=pos, feat_sizes=[(H * 4, W * 4), (H * 2, W * 2), (H, W)],
                                                     output_dict=out, num_frames=n_frames, multiplex_state=mux)
        tap = tracker._prepare_memory_conditioned_features.iou_tap
        n = mux.total_valid_entries
        logits = torch.full((mux.num_buckets, M, 1), -4.0)
        logits.view(-1)[:n] = 6.0
        logits[0, 0, 0] = -4.0 if f in hidden else 5.0
        tap.record(torch.full((mux.num_buckets, M, 3), 0.9), logits)      # what the decoder's call records, as `IoUTap.__call__`
        ptr = torch.zeros(1, M, C)
        ptr[0, :n, :] = float(f)
        out["non_cond_frame_outputs"][f] = {"maskmem_features": torch.randn(1, C, H, W), "maskmem_pos_enc": [torch.zeros(1, C, H, W)],
                                            "image_features": torch.randn(1, H * W, C), "image_pos_enc": torch.zeros(1, H * W, C),
                                            "obj_ptr": ptr, "object_score_logits": mux.demux(logits)}
        if f == add_at:
            mux.add_objects(add_n)
    return out


def memory_scored_per_object():
    # several objects in one call: the bench tool's attach (an explicit focus), since the node keeps core's memory there
    model = _tracked()
    p = _patcher(model)
    L.attach_memory_selection(p)
    p.patch_model(load_weights=False)
    try:
        _group(model.diffusion_model.tracker, n_frames=16, hidden=set(range(3, 13)))
        step = L.get_memory_selection_state(p)
        kept = step.selected_history[15]
        scores = step.frame_scores[8]
    finally:
        p.unpatch_model(unpatch_weights=False)
    assert scores.shape == (3,) and float(scores[0]) <= L.MF_THRESHOLD < float(scores[1]), f"frame 8's scores are {scores.tolist()}"
    assert kept[0] == [1, 2, 13, 14], f"the hidden object kept {kept[0]}: frames 3-12 should be filtered for it alone"
    assert kept[1] == kept[2] == list(range(1, 15)), f"the objects seen throughout kept {kept[1]} and {kept[2]}"
    return f"hidden object keeps {kept[0]}; the two seen keep frames 1-14; frame 8 scores {[round(float(x), 3) for x in scores]}"


def memory_survives_an_object_added():
    """A detection adds an object mid-video (core's `_add_detected_objects`): the frame before it keeps the old count.

    Several objects in one call, so through the bench tool's attach; the node keeps core's memory there."""
    model = _tracked()
    p = _patcher(model)
    L.attach_memory_selection(p)
    p.patch_model(load_weights=False)
    try:
        _group(model.diffusion_model.tracker, n_frames=10, hidden=set(), n_obj=2, add_at=4, add_n=1)
        step = L.get_memory_selection_state(p)
        kept = step.selected_history[9]
        scored = {f: step.frame_scores[f].numel() for f in (4, 5)}
    finally:
        p.unpatch_model(unpatch_weights=False)
    assert scored == {4: 2, 5: 3}, f"frame 4 was scored for {scored[4]} objects and frame 5 for {scored[5]}"
    assert kept[0] == kept[1] == list(range(1, 9)), f"the two first objects kept {kept[0]} and {kept[1]}"
    assert kept[2] == list(range(1, 9)), f"the added object kept {kept[2]}: frames scored before it existed are kept for it, unfiltered"
    # past one bucket of sixteen: the tap's IoU is in the old bucket count, the multiplex state in the new one
    tap = L.IoUTap(lambda *a, **k: None)
    step = L.MemorySelectionStep(tracker=model.diffusion_model.tracker, orig_fn=None, iou_tap=tap)
    tap.last_iou_pred = torch.full((1, 16, 3), 0.9)
    grown = MultiplexState(17, 16, CPU, torch.float32)
    score = step._score_previous_frame(3, {"object_score_logits": torch.full((16, 1), 5.0)}, grown)
    assert score is not None and score.numel() == 16, f"sixteen objects grown to seventeen scored as {score}"
    return f"2 objects, one added after frame 4: scored {scored}; every object keeps frames 1-8; 16 grown to 17 scores 16"


def memory_holds_nothing_after_a_run():
    """After each track returns, nothing on the card is held by the patch and nothing grows from one track to the next."""
    model = _tracked()
    p = _patcher(model)
    L.attach(p, None, image_range=False, text_activation=False)
    step = L.get_memory_selection_state(p)
    p.patch_model(load_weights=False)
    try:
        sizes = []
        for _ in range(4):
            _group(model.diffusion_model.tracker, n_frames=12, hidden=set(range(3, 8)), n_obj=1)
            assert not step.retained_bank, f"{len(step.retained_bank)} frames of memory tensors held after the track returned"
            assert step.iou_tap.last_iou_pred is None and step.iou_tap.last_obj_score_logits is None, "the tap holds the decoder's last output"
            assert all(t.device.type == "cpu" for t in step.frame_scores.values()), "a kept score is not on the CPU"
            sizes.append((len(step._completed_runs), len(step.frame_scores), len(step.selected_history)))
    finally:
        p.unpatch_model(unpatch_weights=False)
    assert len(set(sizes)) == 1 and sizes[0][0] == 0, f"what the patch keeps grew from track to track: {sizes}"
    return f"four tracks: no memory tensors or decoder output held after each, kept (runs, scores, history) {sizes[0]} every time"


def memory_pointers_as_meta():
    """Meta reads an object's pointers from every kept frame but its oldest (video_tracking_multiplex.py:1517)."""
    def handed(hidden: set[int]) -> dict[int, dict[int, bool]]:
        model = _tracked()
        p = _patcher(model)
        L.attach(p, None, image_range=False, text_activation=False)
        step = L.get_memory_selection_state(p)
        seen, stock = {}, step.orig_fn

        def recording(frame_idx, is_init, feats, pos, sizes, output_dict, num_frames, *a, **k):
            seen[frame_idx] = {t: "obj_ptr" in out for t, out in output_dict["non_cond_frame_outputs"].items()}
            return stock(frame_idx, is_init, feats, pos, sizes, output_dict, num_frames, *a, **k)

        step.orig_fn = recording
        p.patch_model(load_weights=False)
        try:
            _group(model.diffusion_model.tracker, n_frames=16, hidden=hidden, n_obj=1)
        finally:
            p.unpatch_model(unpatch_weights=False)
        return seen

    seen = handed(set(range(3, 13)))           # at frame 15 the object keeps 1, 2, 13, 14
    assert seen[15] == {14: True, 13: True, 12: True, 11: False}, f"frame 15 was handed {seen[15]}: the oldest kept frame's pointer is read"
    alone = handed(set(range(1, 16)))          # hidden throughout: only the frame before is kept, every frame
    assert all(v == {f - 1: False} for f, v in alone.items() if f >= 2), f"with only the frame before kept, a pointer was handed: {alone}"
    return "kept 1, 2, 13, 14: pointers from 14, 13, 2 and none from 1; only the frame before kept: no pointer"


def _handed(p, model, **group) -> tuple[dict, dict, L.MemorySelectionStep]:
    """Run `_group` through the patch ComfyUI applied, recording the output_dict core's own method is handed each frame."""
    step = L.get_memory_selection_state(p)
    seen, stock = {}, step.orig_fn

    def recording(frame_idx, is_init, feats, pos, sizes, output_dict, num_frames, *a, **k):
        seen[frame_idx] = output_dict
        return stock(frame_idx, is_init, feats, pos, sizes, output_dict, num_frames, *a, **k)

    step.orig_fn = recording
    p.patch_model(load_weights=False)
    try:
        out = _group(model.diffusion_model.tracker, **group)
    finally:
        p.unpatch_model(unpatch_weights=False)
    return seen, out, step


def memory_multiplex_dms_multi_object():
    """Decoupled Multiplex Memory Selection (`focus_obj_idx=None`): any-visible spatial union + write-time slot gate."""
    hidden = set(range(3, 13))
    # 1. LEGACY CONTROL (`focus_obj_idx=0`): object 0 hidden on 3-12 strips companions 1 and 2's spatial memory.
    model_ctrl = _tracked()
    p_ctrl = _patcher(model_ctrl)
    L.attach_memory_selection(p_ctrl, focus_obj_idx=0)
    seen_ctrl, out_ctrl, _ = _handed(p_ctrl, model_ctrl, n_frames=16, hidden=hidden, n_obj=3)
    two_back_ctrl = seen_ctrl[12]["non_cond_frame_outputs"][10]["maskmem_features"]
    assert two_back_ctrl is out_ctrl["non_cond_frame_outputs"][2]["maskmem_features"], (
        "legacy focus_obj_idx=0 control did not strip companions' spatial memory"
    )

    # 2. NODE DEFAULT (`focus_obj_idx=None, one_object_only=False`): visible companions keep recent spatial memory
    # while hidden object 0 keeps filtering its own pointer history (`[1, 2, 13, 14]` at frame 15).
    model = _tracked()
    p = _patcher(model)
    L.attach(p, None, image_range=False, text_activation=False)
    seen, out, step = _handed(p, model, n_frames=16, hidden=hidden, n_obj=3)
    assert step.focus_obj_idx is None and not step.one_object_only
    two_back_dms = seen[12]["non_cond_frame_outputs"][10]["maskmem_features"]
    assert two_back_dms is out["non_cond_frame_outputs"][10]["maskmem_features"], (
        "DMS spatial union failed to keep recent seen frame 10 for visible companions"
    )
    assert step.selected_history[15][0] == [1, 2, 13, 14], f"object 0 history wrong: {step.selected_history[15][0]}"
    assert step.selected_history[15][1] == step.selected_history[15][2] == list(range(1, 15))
    assert seen[15]["non_cond_frame_outputs"][14]["maskmem_features"] is out["non_cond_frame_outputs"][14]["maskmem_features"]
    # When object 0 has no frame at rank -k (kept length 4 vs union length 14), its obj_ptr slot 0 is zeroed
    # while seen objects 1 and 2 keep their own pointers at rank -5 (frame 10).
    rank5_ptr = seen[15]["non_cond_frame_outputs"][10]["obj_ptr"]
    assert torch.equal(rank5_ptr[0, 0], torch.zeros_like(rank5_ptr[0, 0]))
    assert torch.all(rank5_ptr[0, 1] == 10.0) and torch.all(rank5_ptr[0, 2] == 10.0)

    # Next track of one object on the same patcher runs normally
    p.patch_model(load_weights=False)
    try:
        _group(model.diffusion_model.tracker, n_frames=10, hidden={3}, n_obj=1)
    finally:
        p.unpatch_model(unpatch_weights=False)
    assert 3 not in step.selected_history[9][0], step.selected_history.get(9)

    # 3. WRITE-TIME `MemoryEncodeFilter`: suppress unreliable slots in N_obj >= 2, exempt conditioning / Call #2 / Call #3 / N_obj == 1
    m_enc = _tracked()
    tracker_e = m_enc.diffusion_model.tracker
    p_enc = _patcher(m_enc)
    L.attach(p_enc, None, image_range=False, text_activation=False)
    enc_calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []
    enc_filter = p_enc.object_patches[L.ENCODE_MEM_PATCH]
    assert isinstance(enc_filter, L.MemoryEncodeFilter)

    def rec_encode(pix_feat, pred_masks_high_res, object_score_logits, is_mask_from_pts=False,
                   multiplex_state=None, is_conditioning=False, cond_obj_mask=None):
        enc_calls.append((pred_masks_high_res.clone(), object_score_logits.clone(), bool(is_conditioning)))
        return (
            torch.zeros(multiplex_state.num_buckets, tracker_e.d_model, 8, 8),
            [torch.zeros(multiplex_state.num_buckets, tracker_e.d_model, 8, 8)],
        )

    enc_filter.orig_fn = rec_encode
    step_e = L.get_memory_selection_state(p_enc)
    tap_e = step_e.iou_tap
    pix = torch.zeros(1, tracker_e.d_model, 8, 8)
    p_enc.patch_model(load_weights=False)
    try:
        mux2 = MultiplexState(2, tracker_e.num_multiplex, CPU, torch.float32)
        logits2 = torch.full((1, tracker_e.num_multiplex, 1), -5.0)
        logits2[0, 1, 0] = 6.0  # obj 0 hidden, obj 1 visible
        tap_e.record(torch.full((1, tracker_e.num_multiplex, 3), 0.9), logits2)
        raw_masks = torch.ones(2, 1, 16, 16)
        raw_scores = torch.full((2, 1), 10.0)

        # Call #1: obj 0 suppressed to NO_OBJ_SCORE (-1024) and -10.0; obj 1 untouched
        tracker_e._encode_new_memory(pix, raw_masks, raw_scores, multiplex_state=mux2)
        m1, s1, _ = enc_calls[-1]
        assert enc_filter.suppressed_last_call == [0], enc_filter.suppressed_last_call
        assert torch.all(m1[0] == L.NO_OBJ_SCORE) and torch.all(m1[1] == 1.0)
        assert float(s1[0, 0]) == -10.0 and float(s1[1, 0]) == 10.0

        # Conditioning call: never suppressed
        tracker_e._encode_new_memory(pix, raw_masks, raw_scores, multiplex_state=mux2, is_conditioning=True)
        m_cond, s_cond, _ = enc_calls[-1]
        assert enc_filter.suppressed_last_call == [] and torch.all(m_cond == 1.0) and torch.all(s_cond == 10.0)

        # Call #2 reconditioning (cond_obj_mask=None, same decode_seq, changed slot 0 mask mean) -> obj 0 exempt!
        recond_masks = raw_masks.clone()
        recond_masks[0] = 2.0
        tracker_e._encode_new_memory(pix, recond_masks, raw_scores, multiplex_state=mux2, cond_obj_mask=None)
        m2, s2, _ = enc_calls[-1]
        assert enc_filter.suppressed_last_call == [] and torch.all(m2[0] == 2.0) and float(s2[0, 0]) == 10.0
        assert 0 in enc_filter._reconditioned_slots

        # _score_previous_frame sees reconditioning: boosts score[0] > 0.01 and adds frame to cond_frames
        prev_out_dummy = {
            "pred_masks": torch.zeros(2, 1, 8, 8),
            "object_score_logits": torch.tensor([[-5.0], [6.0]]),
            "ious": torch.full((2, 3), 0.9),
        }
        sc_prev = step_e._score_previous_frame(4, prev_out_dummy, mux2)
        assert sc_prev is not None and float(sc_prev[0]) >= 1.0 and 4 in step_e.cond_frames, (sc_prev, step_e.cond_frames)
        step_e.encode_filter.begin_frame(2)

        # Call #3 birth (_add_detected_objects): slot 0 hidden -> suppressed; slot 2 (newborn & cond_obj_mask=True) -> exempt!
        tap_e.record(torch.full((1, tracker_e.num_multiplex, 3), 0.9), logits2)
        tracker_e._encode_new_memory(pix, raw_masks, raw_scores, multiplex_state=mux2)
        mux3 = MultiplexState(3, tracker_e.num_multiplex, CPU, torch.float32)
        raw_masks3 = torch.ones(3, 1, 16, 16)
        raw_scores3 = torch.full((3, 1), 10.0)
        cond3 = torch.tensor([False, False, True])
        tracker_e._encode_new_memory(pix, raw_masks3, raw_scores3, multiplex_state=mux3, cond_obj_mask=cond3)
        m3, s3, _ = enc_calls[-1]
        assert enc_filter.suppressed_last_call == [0], enc_filter.suppressed_last_call
        assert torch.all(m3[0] == L.NO_OBJ_SCORE) and torch.all(m3[2] == 1.0) and float(s3[2, 0]) == 10.0
        assert enc_filter._had_birth

        # N_obj == 1: NEVER suppressed even when score is -5.0
        mux1 = MultiplexState(1, tracker_e.num_multiplex, CPU, torch.float32)
        logits1 = torch.full((1, tracker_e.num_multiplex, 1), -5.0)
        tap_e.record(torch.full((1, tracker_e.num_multiplex, 3), 0.9), logits1)
        tracker_e._encode_new_memory(pix, raw_masks[:1], raw_scores[:1], multiplex_state=mux1)
        m_single, s_single, _ = enc_calls[-1]
        assert enc_filter.suppressed_last_call == [] and torch.all(m_single == 1.0) and float(s_single[0, 0]) == 10.0
    finally:
        p_enc.unpatch_model(unpatch_weights=False)

    # 4. EXPLICIT `one_object_only=True`: still keeps core's unfiltered memory when N_obj >= 2
    model_ooo = _tracked()
    p_ooo = _patcher(model_ooo)
    L.attach_memory_selection(p_ooo, focus_obj_idx=None, one_object_only=True)
    seen_ooo, out_ooo, step_ooo = _handed(p_ooo, model_ooo, n_frames=12, hidden={3}, n_obj=1, add_at=4, add_n=1)
    assert step_ooo.several_from == 5 and all((d is out_ooo) == (f >= 5) for f, d in seen_ooo.items())

    # 5. MULTI-BUCKET `combined_ptr` (17 objects across 2 buckets, template ptr only 1 bucket): bucket 1 slot 0 preserved
    m17 = _tracked()
    tr17 = m17.diffusion_model.tracker
    step17 = L.MemorySelectionStep(tr17, orig_fn=lambda *a, **k: None, iou_tap=L.IoUTap(lambda *a, **k: None),
                                   focus_obj_idx=None)
    mux17 = MultiplexState(17, tr17.num_multiplex, CPU, torch.float32)
    C17, M17 = tr17.d_model, tr17.num_multiplex
    ptr_f1 = torch.ones(1, M17, C17) * 1.0
    ptr_f2 = torch.ones(2, M17, C17) * 2.0
    ptr_f3 = torch.ones(2, M17, C17) * 3.0
    nco = {
        0: {"maskmem_features": torch.randn(1, C17, 8, 8), "maskmem_pos_enc": [torch.zeros(1, C17, 8, 8)],
            "image_features": torch.randn(1, 64, C17), "image_pos_enc": torch.zeros(1, 64, C17), "obj_ptr": ptr_f1},
        1: {"maskmem_features": torch.randn(1, C17, 8, 8), "maskmem_pos_enc": [torch.zeros(1, C17, 8, 8)],
            "image_features": torch.randn(1, 64, C17), "image_pos_enc": torch.zeros(1, 64, C17), "obj_ptr": ptr_f1},
        2: {"maskmem_features": torch.randn(2, C17, 8, 8), "maskmem_pos_enc": [torch.zeros(2, C17, 8, 8)],
            "image_features": torch.randn(1, 64, C17), "image_pos_enc": torch.zeros(1, 64, C17), "obj_ptr": ptr_f2},
        3: {"maskmem_features": torch.randn(2, C17, 8, 8), "maskmem_pos_enc": [torch.zeros(2, C17, 8, 8)],
            "image_features": torch.randn(1, 64, C17), "image_pos_enc": torch.zeros(1, 64, C17), "obj_ptr": ptr_f3},
    }
    # At rank k=2 -> target_key=2: obj 0 reads f1 (1 bucket) while obj 16 reads f2 (2 buckets).
    valid17_g = [[0, 1, 3]] + [[3]] * 15 + [[1, 2, 3]]
    remapped17 = step17._build_remapped_non_cond(
        frame_idx=4, max_obj_ptrs=10, valid_per_obj=valid17_g, non_cond_outputs=nco,
        multiplex_state=mux17, spatial_valid=[1, 2, 3],
    )
    rank2_ptr = remapped17[2]["obj_ptr"]
    assert rank2_ptr.shape[0] == 2, f"multi-bucket pointer truncated to shape {rank2_ptr.shape}"
    assert torch.all(rank2_ptr[0, 0] == 1.0) and torch.all(rank2_ptr[1, 0] == 2.0), (
        f"bucket 1 slot 0 lost in combined_ptr: b0={rank2_ptr[0, 0, 0].item()}, b1={rank2_ptr[1, 0, 0].item()}"
    )
    text = L.report(p, None)
    assert "Decoupled Multiplex Memory Selection" in text, text
    return "DMS union preserves companions, suppresses bad N>=2 slots, leaves N=1 untouched, and preserves 2-bucket pointers"


def memory_a_later_track_scores():
    """Two tracks with no reset between, the second on an output_dict at the same address starting on a later frame."""
    model = _tracked()
    p = _patcher(model)
    L.attach(p, None, image_range=False, text_activation=False)
    step = L.get_memory_selection_state(p)
    p.patch_model(load_weights=False)
    try:
        out = _group(model.diffusion_model.tracker, n_frames=16, hidden=set(), n_obj=1)
        out["non_cond_frame_outputs"].clear()                  # the same dict object, as a freed dict's address reused
        _group(model.diffusion_model.tracker, n_frames=30, hidden=set(), n_obj=1, out=out, start=20)
        kept = step.selected_history[29]
    finally:
        p.unpatch_model(unpatch_weights=False)
    assert kept == [list(range(20, 29))], f"the second track kept {kept} at frame 29: it ran with its memory gutted"
    return f"the second track keeps frames {kept[0][0]}-{kept[0][-1]} at frame 29"


def memory_left_off_on_sam3():
    p = _patcher(_tracked(SAM3Tracker))
    L.attach(p, None, image_range=False, text_activation=False, memory_selection=True, multimask_ptr=True)
    assert _memory(p) == (None, None), "memory selection was attached to SAM 3's tracker"
    assert L.PROPAGATION_PATCH not in p.object_patches, "multimask_ptr was attached to SAM3Tracker"
    got = L.corrections(p, None)
    assert not got["memory_selection"] and not got["multimask_ptr"] and L.tracker_kind(p) == "SAM3Tracker", got
    text = L.report(p, None)
    assert "memory selection: NOT applied to SAM3Tracker" in text and "SAM 3.1 only" in text, text
    assert "propagation pointer token: NOT applied to SAM3Tracker" in text, text
    assert "memory selection: NOT applied (the model has no video tracker)" in L.report(_patcher(_model()), None)
    return "SAM 3 carries neither memory_selection nor multimask_ptr, and the report says why"


def memory_optional_for_old_graphs():
    # the server refused every graph without the input while it was required (the queue test, 2026-10-18)
    inputs = {i.id: i for i in L.MiniMaxH3SAM31Corrections.define_schema().inputs}
    assert list(inputs.keys())[-2:] == ["correct_multimask_ptr", "correct_detector_presence"], list(inputs.keys())
    mem = inputs["memory_selection"]
    assert mem.optional and mem.default is True, f"memory_selection: optional={mem.optional}, default={mem.default}"
    assert inputs["correct_multimask_ptr"].optional and inputs["correct_multimask_ptr"].default is False
    assert inputs["correct_detector_presence"].optional and inputs["correct_detector_presence"].default is False

    def clip():      # the node clones the pair it is handed; the stand-in CLIP gets a clone like core's
        c = _clip()
        c.clone = lambda: types.SimpleNamespace(patcher=c.patcher.clone(), cond_stage_model=c.cond_stage_model)
        return c

    absent = L.MiniMaxH3SAM31Corrections.execute(_patcher(_tracked()), clip())
    assert isinstance(_memory(absent.result[0])[0], L.MemorySelectionStep), "a graph without the input did not get the default, on"
    got_absent = L.corrections(absent.result[0], absent.result[1])
    assert not got_absent["multimask_ptr"] and not got_absent["detector_presence"], got_absent
    given = L.MiniMaxH3SAM31Corrections.execute(_patcher(_tracked()), clip(), memory_selection=True)
    assert isinstance(_memory(given.result[0])[0], L.MemorySelectionStep), "a graph that sets it on did not get it"
    return "optional inputs at end of schema; memory defaults on, multimask_ptr and detector_presence default off"


def multimask_pointer_matches_best_iou():
    """SAM31Tracker._forward_propagation projects the winning argmax(iou_obj) branch token, not branch 0."""
    def _force_branch(tracker, branch: int, obj_score_bias: float = 5.0):
        iou_lin = [m for m in tracker.sam_mask_decoder.iou_prediction_head.modules() if isinstance(m, torch.nn.Linear)][-1]
        obj_lin = [m for m in tracker.sam_mask_decoder.pred_obj_score_head.modules() if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            iou_lin.bias.fill_(-5.0)
            iou_lin.bias[branch] = 10.0
            obj_lin.bias.fill_(obj_score_bias)
            # `_tracked` scales all params by 0.05; amplify mask_tokens & ptr MLPs so branches stay numerically distinct
            tracker.sam_mask_decoder.mask_tokens.weight.mul_(20.0)
            for mod in (tracker.obj_ptr_proj, tracker.no_obj_ptr_linear):
                for p in mod.parameters():
                    p.mul_(20.0)

    for mem_on, score_bias in ((False, 5.0), (True, 5.0), (False, -5.0)):
        model = _tracked(SAM31Tracker, seed=7)
        tracker = model.diffusion_model.tracker
        _force_branch(tracker, branch=2, obj_score_bias=score_bias)
        mux = MultiplexState(2, tracker.num_multiplex, CPU, torch.float32)
        torch.manual_seed(19)
        feat = torch.randn(1, tracker.d_model, 8, 8)

        captured_hs: list[torch.Tensor] = []
        hook = tracker.sam_mask_decoder.transformer.register_forward_hook(lambda m, i, o: captured_hs.append(o[0].detach()))
        try:
            with torch.no_grad():
                _, _, stock_ptr, score_obj = tracker._forward_propagation(feat, high_res_features=None, multiplex_state=mux)
        finally:
            hook.remove()
        hs_ref = captured_hs[-1]
        M = tracker.sam_mask_decoder.num_multiplex
        T = tracker.sam_mask_decoder.num_mask_output_per_object
        mask_tokens_2d = hs_ref[:, 2 * M : 2 * M + M * T].reshape(hs_ref.shape[0], M, T, -1)
        tokens_obj = mux.demux(mask_tokens_2d)

        def _expected_ptr(branch_idx: int) -> torch.Tensor:
            with torch.no_grad():
                tok = tokens_obj[:, branch_idx]
                p = tracker.obj_ptr_proj(tok)
                if score_bias <= 0:
                    is_obj = (score_obj > 0).float()
                    p = is_obj * p + (1.0 - is_obj) * tracker.no_obj_ptr_linear(p)
                return mux.mux(p)

        exp_0 = _expected_ptr(0)
        exp_2 = _expected_ptr(2)
        assert torch.allclose(stock_ptr, exp_0, atol=1e-5), "stock pointer was not branch 0"
        assert not torch.allclose(exp_0, exp_2, atol=1e-4), "branch 0 and branch 2 pointers coincided"

        p = _patcher(model)
        L.attach(p, None, image_range=False, text_activation=False, memory_selection=mem_on, multimask_ptr=True)
        assert L.corrections(p, None)["multimask_ptr"]
        assert "propagation pointer token: corrected" in L.report(p, None)
        p.patch_model(load_weights=False)
        try:
            with torch.no_grad():
                _, _, fixed_ptr, _ = tracker._forward_propagation(feat, high_res_features=None, multiplex_state=mux)
            tap = p.object_patches[L.MULTIMASK_TAP_PATCH]
            assert tap.last_hs is None and not tap.armed, "MultimaskTokenTap held activation after propagation"
        finally:
            p.unpatch_model(unpatch_weights=False)
        assert torch.allclose(fixed_ptr, exp_2, atol=1e-5), "corrected propagation did not project winning branch 2 token"
        with torch.no_grad():
            _, _, restored_ptr, _ = tracker._forward_propagation(feat, high_res_features=None, multiplex_state=mux)
        assert torch.allclose(restored_ptr, exp_0, atol=1e-5), "unpatch_model did not restore branch 0 pointer"

    # Clone rebinds both tap and propagation wrapper to the clone's own tracker instance
    first, second = _tracked(SAM31Tracker, seed=1), _tracked(SAM31Tracker, seed=2)
    _force_branch(second.diffusion_model.tracker, branch=2, obj_score_bias=5.0)
    p1 = _patcher(first)
    L.attach(p1, None, image_range=False, text_activation=False, memory_selection=False, multimask_ptr=True)
    clone = p1.clone(model_override=(second, ({}, {}, {}, set())))
    prop_c = clone.object_patches[L.PROPAGATION_PATCH]
    tap_c = clone.object_patches[L.MULTIMASK_TAP_PATCH]
    assert prop_c.tracker is second.diffusion_model.tracker and prop_c.orig_fn.__self__ is second.diffusion_model.tracker
    assert tap_c.orig_forward.__self__ is second.diffusion_model.tracker.sam_mask_decoder.transformer
    return "projects winning branch 2 token (with/without memory_selection, positive & negative obj scores), clears last_hs, rebinds clone, undoes cleanly"


def detector_presence_gates_scores():
    """SAM3Detector._detect multiplies sigmoid(presence) into prompt logits in fp32 without fp16 Inf/NaN."""
    class _StubDetector(torch.nn.Module):
        def __init__(self, score_val: float = 3.0, pres_val: float = -5.0):
            super().__init__()
            self.score_val = score_val
            self.pres_val = pres_val

        def _detect(self, features, positions, text_embeddings=None, text_mask=None, points=None, boxes=None):
            B = features[0].shape[0]
            dtype = features[0].dtype
            has_prompts = text_embeddings is not None or points is not None or boxes is not None
            boxes_xyxy = torch.zeros(B, 4, 4, dtype=dtype)
            masks = torch.zeros(B, 4, 8, 8, dtype=dtype)
            scores = torch.full((B, 4), self.score_val, dtype=dtype) if has_prompts else torch.zeros(B, 4, dtype=dtype)
            presence = torch.full((B, 1), self.pres_val, dtype=dtype)
            return boxes_xyxy, scores, masks, {"presence": presence}

    def _det_model(score_val: float = 3.0, pres_val: float = -5.0) -> torch.nn.Module:
        m = _model()
        m.diffusion_model.detector = _StubDetector(score_val=score_val, pres_val=pres_val)
        return m

    for dt in (torch.float32, torch.float16):
        m_absent = _det_model(score_val=10.0, pres_val=-20.0)
        p_absent = _patcher(m_absent)
        L.attach(p_absent, None, image_range=False, text_activation=False, memory_selection=False, detector_presence=True)
        assert L.corrections(p_absent, None)["detector_presence"]
        assert "detector presence head: corrected" in L.report(p_absent, None)
        feats = [torch.zeros(1, 8, 4, 4, dtype=dt)]
        pos = [torch.zeros(1, 8, 4, 4, dtype=dt)]
        prompt = torch.ones(1, 2, 8, dtype=dt)

        _, s_stock, _, _ = m_absent.diffusion_model.detector._detect(feats, pos, text_embeddings=prompt)
        assert float(s_stock.max()) == 10.0

        p_absent.patch_model(load_weights=False)
        try:
            _, s_gated, _, _ = m_absent.diffusion_model.detector._detect(feats, pos, text_embeddings=prompt)
            _, s_unprompted, _, _ = m_absent.diffusion_model.detector._detect(feats, pos)
        finally:
            p_absent.unpatch_model(unpatch_weights=False)

        assert s_gated.dtype == dt and not torch.isnan(s_gated).any() and not torch.isinf(s_gated).any(), s_gated
        expected_p = float(torch.sigmoid(torch.tensor(10.0)) * torch.sigmoid(torch.tensor(-20.0)))
        assert float(s_gated.float().sigmoid().max()) < 1e-4 and abs(float(s_gated.float().sigmoid().mean()) - expected_p) < 2e-3
        assert torch.equal(s_unprompted, torch.zeros_like(s_unprompted)), f"unprompted call altered: {s_unprompted}"

        # Unpatch restores stock scores
        _, s_restored, _, _ = m_absent.diffusion_model.detector._detect(feats, pos, text_embeddings=prompt)
        assert float(s_restored.max()) == 10.0

    # Present concept stays high
    m_present = _det_model(score_val=3.0, pres_val=6.0)
    p_present = _patcher(m_present)
    L.attach(p_present, None, image_range=False, text_activation=False, memory_selection=False, detector_presence=True)
    p_present.patch_model(load_weights=False)
    try:
        _, s_pres, _, _ = m_present.diffusion_model.detector._detect(
            [torch.zeros(1, 8, 4, 4)], [torch.zeros(1, 8, 4, 4)], text_embeddings=torch.ones(1, 2, 8)
        )
        assert float(s_pres.sigmoid().min()) > 0.94, s_pres.sigmoid()
    finally:
        p_present.unpatch_model(unpatch_weights=False)

    # Clone rebinds detector instance cleanly
    second = _det_model(score_val=4.0, pres_val=-6.0)
    clone = p_present.clone(model_override=(second, ({}, {}, {}, set())))
    patch_c = clone.object_patches[L.PRESENCE_PATCH]
    assert patch_c.orig_fn.__self__ is second.diffusion_model.detector
    return "absent presence gates logits strongly in fp32 & fp16 without Inf/NaN, present preserved, unprompted zero, clone & undo verified"


def main() -> int:
    for fn in (the_path_is_cores, a_frame_arrives_in_range, applied_and_undone_by_comfyui, never_twice,
               a_clone_gets_its_own_layer, never_corrects_what_is_right, the_activation_is_swapped, switches_and_report,
               the_input_pair_stays_stock, a_loaded_sibling_changes_nothing, memory_paths_are_cores, memory_on_and_off,
               memory_never_twice, memory_clone_binds_its_own, memory_scored_per_object, memory_left_off_on_sam3,
               memory_survives_an_object_added, memory_holds_nothing_after_a_run, memory_pointers_as_meta,
               memory_optional_for_old_graphs, memory_multiplex_dms_multi_object, memory_a_later_track_scores,
               multimask_pointer_matches_best_iou, detector_presence_gates_scores):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
