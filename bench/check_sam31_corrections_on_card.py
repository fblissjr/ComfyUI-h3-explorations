#!/usr/bin/env python3
"""The SAM 3.1 Corrections node on the real model, on the patcher class the server uses, with a stock pair in the same process.

`bench/check_sam31_corrections.py` shows the patch arithmetic on stand-ins through the plain `ModelPatcher`. This one
answers what that cannot: with dynamic VRAM set up the way `main.py` sets it up (so the loader returns the dynamic
patcher class), does ComfyUI's own loading give each of two patchers that share ONE model the right first layer? A stock
model and its corrected clone share an instance, and `ModelPatcher.clone_has_same_weights` does not compare object
patches, so if ComfyUI ever treats the two as one loaded model the wrong patch state would be served silently.

It loads one SAM 3.1 pair with ComfyUI's loader, makes the corrected clones with the node's own `corrected`, and then
calls ComfyUI's own nodes on each pair in turn, in both orders, reading the range of what reaches the trunk's first
convolution with a hook on that layer, and which activation the text encoder runs while it encodes:

  the_patcher_is_the_servers      the loaded patcher is the dynamic class; otherwise nothing below is about the server.
  detect_in_turn                  `SAM3_Detect` on stock, corrected, stock, corrected, corrected, stock: 0..1 for the
                                  stock pair and exactly -1..1 for the corrected one, every time.
  track_in_turn                   the same through `SAM3_VideoTrack` started from a mask (its resize overshoots, so
                                  stock is about 0..1 and corrected still exactly -1..1).
  trunk_call_in_turn              the same through a direct call of the trunk after `load_model_gpu`, the way the
                                  pack's Subject Track signs a person.
  text_in_turn                    the text encoder runs exact GELU when the corrected clip encodes and the shipped
                                  activation when the stock one does, in turn.
  survives_an_offload             each pair read, everything unloaded (`unload_all_models`), the stock forward seen
                                  back on the model, then each pair read again: still its own range. A full unload
                                  in one process; not the partial eviction a long H3 render provokes on the server.

The frames are made here (a ramp with a block of exact black and one of exact white): nothing depends on what a picture
shows, only on what range reaches the model.

This is NOT the same as a graph on the server's queue, where the executor's cache and its own loading order also take
part; that run is the node's last acceptance. Needs the card and a SAM 3.1 checkpoint; exits 2 without them.

    <comfy venv python> bench/check_sam31_corrections_on_card.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workflows"))

from _lib import card_visible, case, finish, needs  # noqa: E402

needs("a CUDA device (the check is about the server's patcher class on the card)", card_visible())

from h3_config import SEGMENTER  # noqa: E402
from sam3_precision_arms import COMFY, load_model, setup_core  # noqa: E402  (the server-like setup, this folder's own)

torch, mm, comfy, DYNAMIC = setup_core(False, 0)

import torch.nn.functional as F  # noqa: E402

import comfy.clip_model  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402
from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_VideoTrack  # noqa: E402

import sam31_corrections as L  # noqa: E402

PATH = folder_paths.get_full_path("checkpoints", SEGMENTER)
needs(f"the segmenter `{SEGMENTER}` under models/checkpoints", PATH is not None)

STOCK, STOCK_CLIP, _ = load_model(torch, comfy, PATH, None, keep_text_projection=True)
FIXED, FIXED_CLIP = L.corrected(STOCK, STOCK_CLIP)
ORDER = ("stock", "corrected", "stock", "corrected", "corrected", "stock")
PAIRS = {"stock": (STOCK, STOCK_CLIP), "corrected": (FIXED, FIXED_CLIP)}

# A smooth ramp with one block of exact black and one of exact white, so "0..1" and "-1..1" are exact at the first layer
# whatever the resize does between samples; four frames, each shifted a little so the tracker has something to follow.
_ramp = torch.linspace(0.2, 0.8, 128)[None, :, None].expand(96, 128, 3).clone()
_ramp[8:40, 8:40] = 0.0
_ramp[56:88, 88:120] = 1.0
FRAMES = torch.stack([torch.roll(_ramp, shifts=2 * k, dims=1) for k in range(4)])
SEED = torch.zeros((1, 96, 128))
SEED[:, 24:72, 40:88] = 1.0


def _first_layer(model):
    return comfy.utils.get_attr(model.model, L.PATCH_EMBED).proj


def _watch(model, call) -> tuple[float, float]:
    """The lowest and highest value that reached the trunk's first convolution during `call`."""
    seen = []
    hook = _first_layer(model).register_forward_pre_hook(lambda m, i: seen.append((float(i[0].amin()), float(i[0].amax()))))
    try:
        with torch.inference_mode():
            call()
    finally:
        hook.remove()
    assert seen, "nothing reached the first layer"
    return min(s[0] for s in seen), max(s[1] for s in seen)


def _judge(name: str, got: tuple[float, float], slack: float) -> None:
    low, high = got
    if name == "corrected":
        assert abs(low + 1.0) < 1e-3 and abs(high - 1.0) < 1e-3, f"the corrected pair's first layer saw {low:.3f}..{high:.3f}, not -1..1"
    else:
        assert -slack <= low <= slack and 1.0 - slack <= high <= 1.0 + slack, f"the stock pair's first layer saw {low:.3f}..{high:.3f}, not about 0..1"


def _in_turn(run, slack: float) -> str:
    got = []
    for name in ORDER:
        model, clip = PAIRS[name]
        seen = _watch(model, lambda: run(model, clip))
        _judge(name, seen, slack)
        got.append(f"{name} {seen[0]:.2f}..{seen[1]:.2f}")
    return "; ".join(got)


def the_patcher_is_the_servers():
    assert DYNAMIC, "dynamic VRAM could not be set up in this process, so the loader returned the plain patcher class"
    assert isinstance(STOCK, comfy.model_patcher.ModelPatcherDynamic), f"the loaded patcher is {type(STOCK).__name__}"
    assert isinstance(FIXED, comfy.model_patcher.ModelPatcherDynamic), f"the corrected clone is {type(FIXED).__name__}"
    assert FIXED.model is STOCK.model, "the corrected clone has another model instance, so this check is not the shared-instance case"
    assert not any(L.corrections(STOCK, STOCK_CLIP).values()), "the stock pair was patched"
    return f"{type(STOCK).__name__}; the two patchers share one model; corrected: {L.corrections(FIXED, FIXED_CLIP)}"


def detect_in_turn():
    def run(model, clip):
        cond = clip.encode_from_tokens_scheduled(clip.tokenize("person:4"))
        SAM3_Detect.execute(model, FRAMES[:1], conditioning=cond, threshold=0.5, individual_masks=True)
    return _in_turn(run, slack=0.02)


def track_in_turn():
    def run(model, clip):
        SAM3_VideoTrack.execute(FRAMES, model, initial_mask=SEED, conditioning=None, detection_threshold=0.5, max_objects=0, detect_interval=1)
    return _in_turn(run, slack=0.3)       # reasoned: the track node's bicubic resize overshoots a hard edge by a tenth or two; a mapped frame would read -1


def trunk_call_in_turn():
    def run(model, clip):
        mm.load_model_gpu(model)
        x = comfy.utils.common_upscale(FRAMES[:1].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")
        model.model.diffusion_model.detector.backbone["vision_backbone"].trunk(x.to(device=mm.get_torch_device(), dtype=model.model.get_dtype()))
    return _in_turn(run, slack=0.02)


def text_in_turn():
    got = []
    mlp = next(m for m in STOCK_CLIP.cond_stage_model.modules() if isinstance(m, comfy.clip_model.CLIPMLP))
    for name in ORDER:
        clip = PAIRS[name][1]
        ran = []
        hook = mlp.fc1.register_forward_hook(lambda m, i, o: ran.append(mlp.activation is F.gelu))
        try:
            clip.encode_from_tokens_scheduled(clip.tokenize("person"))
        finally:
            hook.remove()
        assert ran, "the text encoder's MLP did not run"
        assert all(ran) == (name == "corrected") and any(ran) == (name == "corrected"), f"the {name} text encoder ran exact GELU: {ran}"
        got.append(f"{name} {'exact' if ran[0] else 'shipped'}")
    return "; ".join(got)


def survives_an_offload():
    """Each pair read, everything unloaded from the card, each pair read again: the patch is on the patcher, not the model."""
    def run(model, clip):
        cond = clip.encode_from_tokens_scheduled(clip.tokenize("person:4"))
        SAM3_Detect.execute(model, FRAMES[:1], conditioning=cond, threshold=0.5, individual_masks=True)

    got = []
    for name in ("corrected", "stock", "corrected"):
        model, clip = PAIRS[name]
        _judge(name, _watch(model, lambda: run(model, clip)), 0.02)
        mm.unload_all_models()
        mm.soft_empty_cache()
        layer = comfy.utils.get_attr(STOCK.model, L.PATCH_EMBED)
        assert not isinstance(getattr(layer, "forward"), L.CorrectedRange), "the patch was still on the model after everything was unloaded"
        seen = _watch(model, lambda: run(model, clip))
        _judge(name, seen, 0.02)
        got.append(f"{name} after a reload {seen[0]:.2f}..{seen[1]:.2f}")
    return "; ".join(got)


def main() -> int:
    print(f"ComfyUI at {COMFY.name}, dynamic VRAM {'on' if DYNAMIC else 'OFF'}, patcher {type(STOCK).__name__}")
    for fn in (the_patcher_is_the_servers, detect_in_turn, track_in_turn, trunk_call_in_turn, text_in_turn, survives_an_offload):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
