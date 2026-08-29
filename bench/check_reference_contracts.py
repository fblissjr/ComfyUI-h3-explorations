#!/usr/bin/env python3
"""The load-bearing contracts in core's reference node, asserted for the first time.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). CPU only, no
server, no weights: both VAEs are stubs and the node is driven only as far as
the point where its two reference lists are complete.

**Why this exists.** `docs/research/conditioning_nodes.md` records five
load-bearing contracts in `MiniMaxH3ReferenceToVideo`, plus two smaller ones,
and says of all of them: they fail silently and are **enforced by nothing**.
That is `CLAUDE.md`'s "a requirement is not a control" case, seven times over,
and it has stood through two postmortems. This file is the control.

It is also the prerequisite for replacing that node. A replacement has to
reproduce every one of these, and until something asserts them "did we
reproduce them" is unanswerable. **Core is the control here, not a fixture**:
every expectation below is read off core's own behaviour on inputs built to
make the contract's failure mode visible, rather than compared against a
recorded blob that would freeze today's bugs as tomorrow's expectations.

**How it observes.** `clip.tokenize` is patched. At the moment core calls it,
both `ref_items` and `ref_blocks` are complete and still live in the caller's
frame, so the spy reads them out of `sys._getframe` and raises. That is a
reach into a private frame and it is deliberate: the alternative is asserting
against the conditioning tensor, which needs a 32B encoder and would test the
DiT's packing rather than the node's bookkeeping.

**All seven are covered as of 2026-08-22**, contracts 4 and 5 last. Both had
been recorded here as out of reach -- 4 as `model_base.py`'s job rather than
this node's, 5 as needing a loaded 32B encoder -- and neither held.
`MiniMaxH3.extra_conds` runs on a stub supplying `concat_keys` and
`model_config`; `CLIP.encode_from_tokens_scheduled` runs over a stub text
encoder returning the same 3-tuple the real one does. The blocker was where to
point the harness, not what it would cost.

**One gap in core is measured and deliberately NOT failed here.** With forced
CLIP-schedule hooks active, `comfy/sd.py:379-381` reads only `o[:2]` and never
`o[2]`, so `minimax_token_tags` is dropped. Executed 2026-08-22 against a stub
with `forced_hooks` set: the conditioning extras come back as
`clip_start_percent`, `clip_end_percent`, `pooled_output` and nothing else.
Case 5c records it rather than failing on it -- it is core's behaviour, no
shipped graph here wires CLIP hooks, and a check that reds on a correct state
trains readers to ignore red. **If 5c ever flips, upstream fixed it and the
case should be retired rather than repaired.**

Exit 0 all covered contracts hold, 1 one is violated, 2 the harness could not
reach the node's bookkeeping at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ComfyUI first, repo dir NOT on the path: `nodes_minimax_h3` does a bare
# `import nodes` and this repo's own nodes.py wins from position 0.
_COMFY = Path.home() / "ComfyUI"
sys.path.insert(0, str(_COMFY))

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True

VISION_START = 151652
VISION_END = 151653


class _Reached(Exception):
    """Raised once both reference lists are complete."""


class _StubVideoVae:
    """Returns a latent of the right rank; nothing here reads its values."""

    def encode(self, frames):
        import torch
        t = int(frames.shape[0])
        h, w = int(frames.shape[1]), int(frames.shape[2])
        return torch.zeros(1, 24, max(1, t // 4 + 1), h // 16, w // 16)


class _StubAudioVae:
    audio_sample_rate = 32000

    def encode(self, waveform):
        import torch
        # [1, 32, 2, T]; T distinct per call so a mispairing is visible
        n = int(waveform.shape[-2])
        return torch.zeros(1, 32, 2, max(1, n // 640))


def _audio(seconds: float, sr: int = 32000):
    import torch
    return {"waveform": torch.zeros(1, 2, int(sr * seconds)),
            "sample_rate": sr}


def _drive(core, **kwargs):
    """Run core's node until its two reference lists are complete.

    Returns (ref_items, ref_blocks) read from the caller's frame at the moment
    `clip.tokenize` is called -- the first point where both are finished.
    """
    captured = {}

    class _SpyClip:
        def tokenize(self, _prompt, **_kw):
            frame = sys._getframe(1)
            captured["items"] = frame.f_locals.get("ref_items")
            captured["blocks"] = frame.f_locals.get("ref_blocks")
            raise _Reached

    try:
        core.MiniMaxH3ReferenceToVideo.execute(
            clip=_SpyClip(), vae=_StubVideoVae(), audio_vae=_StubAudioVae(),
            prompt="a reference prompt", width=1344, height=768, length=124,
            **kwargs)
    except _Reached:
        pass
    if captured.get("items") is None or captured.get("blocks") is None:
        raise RuntimeError(
            "could not read ref_items/ref_blocks out of core's frame; its "
            "local names changed and this harness is asserting nothing")
    return captured["items"], captured["blocks"]


def contract4_holds():
    """(ok, detail) for: keyframe latents precede reference latents.

    A callable rather than inline in main() so `bench/red/` can drive it with
    the subject mutated. Marker VALUES, not counts -- a reversal has to be
    unambiguous, and two same-length lists in the wrong order are not.
    """
    import torch
    pl = _drive_extra_conds(torch.full((1,), 7.0), torch.full((1,), 9.0),
                            torch.full((1,), 70.0), torch.full((1,), 90.0))
    vids = [float(t[0]) for t in pl["cond_video_latents"]]
    auds = [float(t[0]) for t in pl["cond_audio_latents"]]
    return (vids == [7.0, 9.0] and auds == [70.0, 90.0],
            f"video {vids}, audio {auds} -- keyframe marker must come first "
            f"in both")


def contract5a_holds():
    """(ok, detail) for: the tags reach conditioning through the path the node uses.

    Asserted end-to-end at `encode_from_tokens_scheduled`, the node's own
    entry point, and by VALUE rather than by key presence -- see the detail
    string for why those are different assertions.
    """
    import torch
    tags = torch.tensor([0, 1, 0, 1])
    out = _drive_encode_from_tokens({"minimax_token_tags": tags})
    got = (out[0][1].get("minimax_token_tags")
           if isinstance(out, list) and out and isinstance(out[0][1], dict)
           else None)
    ok = isinstance(got, torch.Tensor) and torch.equal(got, tags)
    return (ok,
            f"conditioning carries {list(got) if isinstance(got, torch.Tensor) else got!r} "
            f"against the marker {list(tags)} -- **value equality, not key "
            f"presence**: a key holding None or an all-zero tensor is exactly "
            f"the silent 'every row is text' failure this contract exists to "
            f"prevent, and both pass a presence test")


def contract5b_holds():
    """(ok, detail) for: losing the tags is silent, which is why this exists."""
    import torch
    z = torch.zeros(1)
    marker = torch.tensor([0, 1, 0, 1])
    with_tags = _drive_extra_conds(z, z, z, z, tags=marker)
    without = _drive_extra_conds(z, z, z, z)
    got = with_tags.get("text_token_tags")
    carried = isinstance(got, torch.Tensor) and torch.equal(got, marker)
    return (carried and "text_token_tags" not in without,
            f"payload carries {list(got) if isinstance(got, torch.Tensor) else got!r} "
            f"against the marker {list(marker)}, and the absent case raised "
            f"nothing. **Value equality, not presence** -- an all-zero tensor "
            f"reaching the DiT tags every row as text while passing any "
            f"presence test")


def _drive_extra_conds(keyframe_latent, ref_latent,
                       keyframe_audio, ref_audio, tags=None):
    """Run core's real `MiniMaxH3.extra_conds` and return its payload.

    Contract 4 lives in `comfy/model_base.py`, not in the node, so this is the
    one assertion here whose subject is the model rather than
    `MiniMaxH3ReferenceToVideo`. The stub supplies only what `BaseModel`
    touches on the path taken with `cross_attn=None` -- `concat_keys` and
    `model_config` -- so the ordering statements execute unmodified.
    """
    import comfy.model_base as mb

    class _Probe(mb.MiniMaxH3):
        def __init__(self):
            self.concat_keys = ()
            self.model_config = None

        def audio_scale(self):
            return 1.0

    kw = dict(device="cpu", cross_attn=None,
              minimax_keyframes=[{"latent": keyframe_latent,
                                  "audio_latent": keyframe_audio}],
              minimax_refs=[{"latent": ref_latent,
                             "audio_latent": ref_audio}])
    if tags is not None:
        kw["minimax_token_tags"] = tags
    out = _Probe().extra_conds(**kw)
    if "minimax_payload" not in out:
        raise RuntimeError(
            "extra_conds returned no `minimax_payload`; the key was renamed "
            "and this harness is asserting nothing")
    return out["minimax_payload"].cond


def _drive_encode_from_tokens(extra: dict, hooked: bool = False):
    """Run core's real encode path over a stub text encoder.

    Contract 5's subject is `comfy/sd.py`'s merge of the encoder's third
    return value into the conditioning dict. The stub returns a 3-tuple
    exactly as `MiniMaxH3ClipModel.encode_token_weights` does; everything that
    decides whether the third element survives is core's.

    **The normal branch drives `encode_from_tokens_scheduled`, which is what
    the node actually calls** (`comfy_extras/nodes_minimax_h3.py`:
    `clip.encode_from_tokens_scheduled(tokens)`). Driving
    `encode_from_tokens` directly with an explicit `return_dict=True` --
    which this did until it was audited on 2026-08-22 -- asserts that the
    merge works when asked for, and says nothing about whether the production
    caller asks. That call site cannot silently regress -- flipping it raises
    `AttributeError: 'tuple' object has no attribute 'pop'` on the next line,
    executed rather than reasoned -- but the earlier framing was still testing
    the mechanism instead of the seam, so the drive moved to the end-to-end
    path regardless. **The direct `return_dict=False` arm went with it rather
    than being kept as documentation**: it had no caller once 5a asserted
    value equality, and a future change to that unused return shape must not
    fail a healthy production seam. Inert code that still runs is a failure
    this repo has recorded before.
    """
    import comfy.sd
    import torch

    class _StubTE:
        def set_clip_options(self, _o):
            pass

        def reset_clip_options(self):
            pass

        def encode_token_weights(self, _t):
            return torch.zeros(1, 3, 8), torch.zeros(1, 8), dict(extra)

    class _StubCLIP:
        cond_stage_model = _StubTE()
        layer_idx = None
        apply_hooks_to_conds = None
        use_clip_schedule = False

        class patcher:
            load_device = torch.device("cpu")
            forced_hooks = None

        def load_model(self, _t):
            pass

        # Delegated to core rather than stubbed: these two are ON the path
        # under test, and a local stand-in would test the stand-in.
        def add_hooks_to_dict(self, d):
            return comfy.sd.CLIP.add_hooks_to_dict(self, d)

        def encode_from_tokens(self, *a, **kw):
            return comfy.sd.CLIP.encode_from_tokens(self, *a, **kw)

    class _Hooks:
        def get_hooks_for_clip_schedule(self):
            return [((0.0, 1.0), [])]

        def reset(self):
            pass

    clip = _StubCLIP()
    if hooked:
        # The OTHER branch of encode_from_tokens_scheduled, taken when the
        # patcher carries forced hooks. Reached by giving the stub patcher a
        # hooks object rather than by mutating core.
        class _HookedPatcher:
            load_device = torch.device("cpu")
            forced_hooks = _Hooks()

            def patch_hooks(self, _h):
                pass

        clip.patcher = _HookedPatcher()
        clip.use_clip_schedule = True
        return comfy.sd.CLIP.encode_from_tokens_scheduled(clip, "tokens",
                                                          show_pbar=False)
    return comfy.sd.CLIP.encode_from_tokens_scheduled(clip, "tokens")


def contract5c_state():
    """(ok, detail) for CORE's forced-CLIP-schedule branch, which drops the tags.

    **Not a failure of anything in this repo, and it must not red.**
    `comfy/sd.py:379-381` reads `o[:2]` and never `o[2]`, so with forced hooks
    active `minimax_token_tags` never reaches conditioning and the DiT tags
    every row as text. No graph here wires CLIP hooks, so the shipped path is
    the no-hook branch that case 5a covers.

    Recorded rather than fixed, and asserted in its CURRENT state so the
    record cannot rot: `ok` is True while the tags are still dropped. **If
    this flips, upstream fixed it** -- retire the case and the gap entry,
    do not repair them.

    That contract was carried out on 2026-08-29 for its previous exemplar,
    `bench/check_mono_ref_audio.py`, and the way it went wrong is worth
    inheriting here. That gate asserted a mono reference raises, and it
    verified the claim against a hand-built 1-channel latent rather than
    against a real encode -- so when core started upmixing mono one wrapper
    above the layer the gate traced, the gate stayed green and the fix went
    unnoticed for a week. **A current-state assertion is only as good as its
    entry point.** This case drives `encode_from_tokens_scheduled` itself,
    which is the seam the claim is about.
    """
    import torch
    out = _drive_encode_from_tokens({"minimax_token_tags": torch.tensor([0, 1, 0])},
                                    hooked=True)
    extras = sorted(out[0][1]) if isinstance(out, list) and out else out
    dropped = isinstance(out, list) and "minimax_token_tags" not in out[0][1]
    return (dropped,
            f"forced-hook branch yields {extras} -- the tags are "
            f"{'dropped, as core still does' if dropped else 'PRESENT, so upstream fixed it: retire this case'}")


def _labels(ref_items):
    """The presentation core's tokenizer builds, decoded, with vision marked."""
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    tok = MiniMaxH3Tokenizer()
    entries = tok.tokenize_with_weights(
        "", minimax_ref_items=ref_items)["qwen3vl_32b"][0]
    hf = tok.qwen3vl_32b.tokenizer
    out, run = [], []
    for t, *_ in entries:
        if isinstance(t, int) and t not in (VISION_START, VISION_END):
            run.append(t)
            continue
        if run:
            out.append(hf.decode(run))
            run = []
        out.append("<VISION_START>" if t == VISION_START
                   else "<VISION_END>" if t == VISION_END else "<BLOCK>")
    if run:
        out.append(hf.decode(run))
    return out


def main() -> int:
    try:
        import torch  # noqa: F401
        import comfy_extras.nodes_minimax_h3 as core
    except Exception as exc:
        print(f"could not import core: {exc}")
        print("nothing was checked")
        return 2

    results = []

    def record(name, ok, detail):
        results.append((name, ok, detail))
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}\n        {detail}")

    print("core's reference-node contracts, asserted against core itself\n")

    try:
        # --- Contract 1: the two lists are NOT index-aligned ---------------
        items, blocks = _drive(
            core,
            ref_videos={"ref_video_0": torch.zeros(22, 544, 960, 3)},
            ref_video_audios={"ref_video_audio_0": _audio(1.0)},
        )
        record("contract 1: ref_items and ref_blocks are not index-aligned",
               len(items) != len(blocks),
               f"one sounded video gives {len(items)} presentation item(s) "
               f"and {len(blocks)} DiT block(s). Index-aligning them mislabels "
               f"every reference after the first sounded video.")

        # --- Contract 2: soundtracks pair by socket-name SUFFIX ------------
        # Two videos of different lengths, and the dict deliberately supplies
        # the soundtracks in the opposite order to the videos. Pairing by
        # position would attach each track to the wrong video; pairing by
        # suffix attaches them correctly. The audio stub's row count differs
        # per track, so the mispairing would be visible in the blocks.
        items, blocks = _drive(
            core,
            ref_videos={"ref_video_0": torch.zeros(22, 544, 960, 3),
                        "ref_video_1": torch.zeros(22, 544, 960, 3)},
            ref_video_audios={"ref_video_audio_1": _audio(4.0),
                              "ref_video_audio_0": _audio(1.0)},
        )
        sounded = [b for b in blocks if b.get("kind") == "video_audio"]
        rows = [b["ref_audio_t"] for b in sounded]
        # video_0 gets the 1.0s track, video_1 the 4.0s one, in block order
        paired_right = len(rows) == 2 and rows[0] < rows[1]
        record("contract 2: soundtracks pair by socket-name suffix",
               paired_right,
               f"soundtracks supplied in reverse order still attach by name: "
               f"block audio rows in video order are {rows}, ascending because "
               f"ref_video_0 got the shorter track. Positional pairing would "
               f"reverse this.")

        # --- Contract 3: <Audio j> is one shared counter, standalone last ---
        items, blocks = _drive(
            core,
            ref_images={"ref_image_0": torch.zeros(1, 768, 768, 3)},
            ref_videos={"ref_video_0": torch.zeros(22, 544, 960, 3)},
            ref_video_audios={"ref_video_audio_0": _audio(1.0)},
            ref_audios={"ref_audio_0": _audio(2.0)},
        )
        # Filter the SENTINEL markers only. Filtering anything starting with
        # "<" also removes "<Audio 1>: " and "<Video 1>: ", which are the very
        # strings being asserted -- the first version of this line did exactly
        # that and the contract could not be found in its own evidence.
        _MARK = {"<VISION_START>", "<VISION_END>", "<BLOCK>"}
        text = "".join(x for x in _labels(items) if x not in _MARK)
        shared = "<Audio 1>" in text and "<Audio 2>" in text
        order = (text.index("<Audio 1>") < text.index("<Video 1>")
                 < text.index("<Audio 2>"))
        record("contract 3: one <Audio j> counter, standalone audio last",
               shared and order,
               f"the video's soundtrack takes <Audio 1> immediately before "
               f"<Video 1>, and the standalone track takes <Audio 2> after it. "
               f"A prompt saying <Audio 1> means a different thing depending "
               f"on whether a soundtrack is wired.")

        # --- Small contract: vision sentinels flank EVERY block -------------
        seq = _labels(items)
        pairs_ok = True
        for i, tokname in enumerate(seq):
            if tokname == "<BLOCK>":
                pairs_ok &= (i > 0 and seq[i - 1] == "<VISION_START>"
                             and i + 1 < len(seq) and seq[i + 1] == "<VISION_END>")
        nblocks = seq.count("<BLOCK>")
        record("small contract: vision sentinels flank every vision block",
               pairs_ok and nblocks > 0,
               f"{nblocks} vision block(s), each between <|vision_start|> and "
               f"<|vision_end|>. Without both, two rows per block get tagged "
               f"as text in the DiT.")

        # --- Small contract: prompt weighting stays disabled ----------------
        from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
        tok = MiniMaxH3Tokenizer()
        weighted = tok.tokenize_with_weights("(a cat:1.5) on a mat")
        ws = {float(e[1]) for e in weighted["qwen3vl_32b"][0]}
        record("small contract: prompt weighting is disabled",
               ws == {1.0},
               f"a CLIP-style weight in the prompt yields weights {sorted(ws)}. "
               f"Anything but 1.0 applies a blend to a hidden state this model "
               f"was never trained with.")

    except RuntimeError as exc:
        print(f"\n{exc}")
        print("nothing was checked")
        return 2
    except Exception as exc:
        print(f"\ncould not drive core's node: {type(exc).__name__}: {exc}")
        print("nothing was checked")
        return 2

    # ---- contract 4: keyframe latents precede reference latents ----
    #
    # Established by STATEMENT ORDER in extra_conds and nothing else: the
    # keyframe block seeds cond_video_latents and the refs block appends to
    # it. Swapping the two blocks reverses the flat lists and raises nothing.
    # Marker values rather than counts, so a reversal is unambiguous.
    try:
        record("contract 4  keyframe latents precede reference latents",
               *contract4_holds())
    except Exception as exc:
        record("contract 4  keyframe latents precede reference latents", False,
               f"could not drive extra_conds: {type(exc).__name__}: {exc}")

    # ---- contract 5: the tags ride the return_dict=True path, silently ----
    #
    # Two failures, and the second is why this needs a control at all: losing
    # the dict path drops the tags, and losing the tags raises nothing -- the
    # DiT simply tags every row as text.
    try:
        record("contract 5a return_dict=True carries the encoder's extras",
               *contract5a_holds())
        record("contract 5b dropping the tags is SILENT, not an error",
               *contract5b_holds())
        record("contract 5c core drops the tags under forced CLIP hooks "
               "(recorded, not owned)", *contract5c_state())
    except Exception as exc:
        record("contract 5  minimax_token_tags reaches conditioning", False,
               f"could not drive the path: {type(exc).__name__}: {exc}")

    print("\nEvery contract in docs/research/conditioning_nodes.md now has an "
          "assertion.\nContracts 4 and 5 are asserted against "
          "comfy/model_base.py and comfy/sd.py\nrather than against the node, "
          "because that is where they live.")

    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f"\nFAIL: {len(failed)} contract(s) violated: {failed}")
        return 1
    print(f"\nok    {len(results)} contract(s) hold, asserted against core")
    return 0


if __name__ == "__main__":
    sys.exit(main())
