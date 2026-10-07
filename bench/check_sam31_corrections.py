#!/usr/bin/env python3
"""The SAM 3.1 Corrections node on stand-in modules, through ComfyUI's real ModelPatcher. No SAM weights, no card.

`sam31_corrections.py` attaches two corrections to a loaded SAM 3.1 pair as object patches: the image
range at the trunk's patch embedding, and the text encoder's activation. This check builds a stand-in with the same
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
  switches_and_report             each correction alone, neither, and both: `corrections` reads exactly what was
                                  attached, and the report says so in words, naming what is never corrected.

What it cannot check: the server's dynamic patcher class (this builds the plain `ModelPatcher`), the real model, and
that ComfyUI's own SAM nodes reach the patched layer. That is a real-model run as a graph on the server's queue.

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


def main() -> int:
    for fn in (the_path_is_cores, a_frame_arrives_in_range, applied_and_undone_by_comfyui, never_twice,
               a_clone_gets_its_own_layer, never_corrects_what_is_right, the_activation_is_swapped, switches_and_report,
               the_input_pair_stays_stock):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
