#!/usr/bin/env python3
"""Does `MiniMaxH3Conditioning` still do the things it exists to do?

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). CPU only, no
server, no weights: the VAE is a stub and `clip.tokenize` is patched so the
node is driven to the point where its geometry decisions are complete and no
further.

**Why this exists.** `MiniMaxH3Conditioning` replaces core's
`MiniMaxH3ImageToVideo` for t2va and the keyframe signatures, and it owns 19
shipped graphs. Its correctness has been asserted in a docstring and controlled
by nothing, carried as a forward item through **three** postmortems. This is
the control.

**Core is the reference, and the point is that it is right in both
directions.** A replacement node has two ways to be wrong: it can break
something core got right, or it can quietly stop doing the thing it was built
for. So every arm below is one of:

  AGREE   ours must match core, because this is a replacement and not a rewrite
  DIFFER  ours must NOT match core, because a documented defect is being fixed

A DIFFER arm that starts agreeing is the more dangerous failure and the one
nothing would otherwise notice -- the node keeps running, the graphs keep
rendering, and the fix is silently gone. That is why the fixes are asserted as
differences from core rather than against remembered numbers.

**What is NOT covered, named rather than skipped.** The conditioning tensor
itself. Everything here is the node's bookkeeping -- geometry, keyframe
pinning, refusal, marker routing -- observed at the point of the tokenize call.
Whether the encoder then produces the right hidden states is
`bench/grade_h3_marker_tokens.py`'s question and needs a loaded 32B encoder.

Exit 0 all arms behave, 1 an arm does not, 2 the harness could not drive
either node.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
# ComfyUI first, repo dir NOT on the path: `nodes_minimax_h3` does a bare
# `import nodes` and this repo's own nodes.py wins from position 0. The pack is
# reached as a package instead.
_COMFY = Path.home() / "ComfyUI"
sys.path.insert(0, str(_COMFY))
sys.path.insert(0, str(_REPO.parent))
_PKG = _REPO.name

MARKER_PROMPT = "She says, <d>[English] We leave now.</d>"


class _Reached(Exception):
    """Raised once the node's geometry decisions are complete."""


class _StubVae:
    def encode(self, image):
        import torch
        h, w = int(image.shape[1]), int(image.shape[2])
        return torch.zeros(1, 24, 1, h // 16, w // 16)


class _SpyClip:
    """Records the tokenize call and the caller's geometry, then stops."""

    def __init__(self, store):
        self.store = store
        # a real tokenizer, so marker routing is observable
        from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
        self._tok = MiniMaxH3Tokenizer()

    def clone(self):
        return self

    @property
    def tokenizer(self):
        return self._tok

    def tokenize(self, prompt, **kwargs):
        frame = sys._getframe(1)
        loc = frame.f_locals
        self.store["width"] = loc.get("width")
        self.store["height"] = loc.get("height")
        self.store["images"] = loc.get("images")
        self.store["keyframes"] = loc.get("keyframes")
        entries = self._tok.tokenize_with_weights(prompt, **kwargs)
        self.store["ids"] = [t[0] for t in entries["qwen3vl_32b"][0]
                             if isinstance(t[0], int)]
        raise _Reached


def _img(w, h):
    import torch
    return torch.zeros(1, h, w, 3)


def _drive(node, **kwargs):
    """(captured, raised) for one node call."""
    store = {}
    try:
        node.execute(clip=_SpyClip(store), vae=_StubVae(), **kwargs)
    except _Reached:
        return store, None
    except Exception as exc:
        return store, exc
    return store, None


def main() -> int:
    try:
        import comfy_extras.nodes_minimax_h3 as core
        import importlib
        ours = importlib.import_module(f"{_PKG}.conditioning").MiniMaxH3Conditioning
    except Exception as exc:
        print(f"could not import core or the pack: {exc}")
        print("nothing was checked")
        return 2

    results = []

    def record(kind, name, ok, detail):
        results.append((name, ok))
        print(f"  {'ok  ' if ok else 'FAIL'}  [{kind}] {name}\n        {detail}")

    print("MiniMaxH3Conditioning against core's MiniMaxH3ImageToVideo\n")

    base = dict(prompt="a woman walks toward the camera",
                width=1344, height=768, length=124)

    # --- AGREE: plain t2va geometry must match core -----------------------
    mine, err_m = _drive(ours, **base)
    theirs, err_t = _drive(core.MiniMaxH3ImageToVideo, **base)
    if err_m or err_t or not mine or not theirs:
        print(f"could not drive both nodes on the plain case: "
              f"ours={err_m!r} core={err_t!r}")
        print("nothing was checked")
        return 2
    same = (mine["width"], mine["height"]) == (theirs["width"], theirs["height"])
    record("AGREE", "plain t2va geometry matches core", same,
           f"ours {mine['width']}x{mine['height']}, core "
           f"{theirs['width']}x{theirs['height']}. A replacement that moved "
           f"the default canvas would change every shipped t2va graph.")

    # --- DIFFER: an empty prompt is refused -------------------------------
    _, err_m = _drive(ours, **{**base, "prompt": "   "})
    _, err_t = _drive(core.MiniMaxH3ImageToVideo, **{**base, "prompt": "   "})
    refused = isinstance(err_m, ValueError)
    core_renders = not isinstance(err_t, ValueError)
    record("DIFFER", "an empty prompt is refused, where core conditions on a pad",
           refused and core_renders,
           f"ours raised {type(err_m).__name__ if err_m else 'nothing'}; core "
           f"raised {type(err_t).__name__ if err_t else 'nothing'} and would "
           f"render against token 151643.")

    # --- DIFFER: last-frame-only keeps its whole picture ------------------
    # A last frame whose aspect differs from the default canvas. Core picks
    # geometry from which socket was wired and cover-crops it into a canvas
    # chosen elsewhere; ours anchors on the frame that was actually supplied,
    # so the canvas follows the picture instead of cropping it.
    last = _img(768, 1344)                       # portrait, against a landscape default
    mine, _ = _drive(ours, **{**base, "last_frame": last})
    theirs, _ = _drive(core.MiniMaxH3ImageToVideo, **{**base, "last_frame": last})
    mine_wh = (mine.get("width"), mine.get("height"))
    theirs_wh = (theirs.get("width"), theirs.get("height"))
    portrait = mine_wh[0] is not None and mine_wh[1] > mine_wh[0]
    record("DIFFER", "last-frame-only anchors on the frame that was supplied",
           mine_wh != theirs_wh and portrait,
           f"ours {mine_wh[0]}x{mine_wh[1]} follows the portrait keyframe; "
           f"core {theirs_wh[0]}x{theirs_wh[1]} keeps a canvas chosen "
           f"elsewhere and crops into it. If these ever match, the fix is gone "
           f"and nothing else would say so.")

    # --- AGREE: only the wired keyframe is pinned -------------------------
    mine, _ = _drive(ours, **{**base, "last_frame": last})
    kfs = mine.get("keyframes") or []
    idxs = [k["resolved_frame_index"] for k in kfs]
    record("AGREE", "only the wired keyframe is pinned",
           len(kfs) == 1 and idxs[0] != 0,
           f"one keyframe at index {idxs[0] if idxs else None}, the target's "
           f"final frame. Geometry resolution fills both slots with the anchor, "
           f"which is right for a canvas answer and wrong for a row map: "
           f"pinning an absent first frame would anchor frame 0 on a picture "
           f"nobody supplied.")

    # --- AGREE: the dialogue marker reaches the encoder as one id ---------
    mine, _ = _drive(ours, **{**base, "prompt": MARKER_PROMPT})
    ids = mine.get("ids") or []
    record("AGREE", "a dialogue marker routes to its real token id",
           151669 in ids,
           f"<d> arrives as id 151669 rather than as BPE debris. Inherited "
           f"from the tokenizer since the core fix, and asserted here because "
           f"this node used to be the only thing supplying it and still calls "
           f"clip_with_vendor_tokens for installs without that fix.")

    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\nFAIL: {len(failed)} arm(s): {failed}")
        return 1
    print(f"\nok    {len(results)} arm(s) behave, against core")
    return 0


if __name__ == "__main__":
    sys.exit(main())
