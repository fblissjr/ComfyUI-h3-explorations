#!/usr/bin/env python3
"""`MiniMaxH3FrozenVideoCache` on a tiny H3 model: which path each call takes, and what it computes.

`frozen_video_cache.py` is the node. It changes what the model computes on a
cached step, so this runs core's real `MiniMaxH3Model` (three blocks, random
weights, bf16 compute on CPU) through the node's own patches and wrappers, the
way the sampler would, and grades:

1. **The path taken, by count.** A build call runs every block through the
   build path and a cached call runs every block cached. The gate sends a
   call with no mask, a partly open video (0.05) or a frozen audio stock. A
   check that only compared outputs would pass vacuously on a call that never
   reached the cache.
2. **The build is the stock forward**, bit for bit, because it runs the stock
   block. Its output must match the stock model call exactly.
3. **A cached step with nothing moved matches stock.** The same inputs as the
   build, so every cached row is exactly what the stock forward would compute,
   under the bf16 codec, which is lossless on bf16 compute. What is left is
   kernel order on a shorter query.
4. **The control that proves the check can see the approximation.** Move the
   audio and the timestep, and the cached step must now differ from stock by
   clearly more than item 3's floor: the cached rows no longer see the new
   audio. It must still be closer to that stock step than to the one it was
   built from, which shows the live rows are live.
5. **Text rows are live, and that matters.** With the timestep moved, the
   cached step with text live must be closer to stock than the same step with
   text cached (`LIVE_KINDS` monkeypatched to audio alone).
6. **Rebuild and refusal.** A timestep past the cond pin rebuilds. A foreign
   `patches_replace["dit"]` entry is refused when the node attaches, and an
   object-patched attention forward sends the call stock with a reason. The
   OUTER_SAMPLE wrapper frees the store when the pass ends.
7. **The codecs round-trip within their bounds**, and `verify` records the
   cosine it promises.

**What this does NOT establish:** anything on the card. These stay unverified
until a live run: the kitchen backend with a query shorter than the keys, the
full-sequence qkv transient in the cached step, the compiler flip, wall time,
and what the approximation costs on real audio. The node's `verify` switch is
the first live run's instrument.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_frozen_video_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(COMFY))

import torch  # noqa: E402

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True
import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.ldm.minimax.model as mm_h3  # noqa: E402
from comfy.patcher_extension import WrappersMP  # noqa: E402
import frozen_video_cache as fvc  # noqa: E402

HIDDEN = 256
N_BLOCKS = 3
TEXT_LEN = 8
LATENT = (3, 4, 4)       # latent t, h, w
AUDIO_T = 10
#: Item 3's bound. **Reasoned**: bf16 carries 8 mantissa bits, so a matching
#: forward agrees to a few 1e-3 in relative L2; a severed edge moves it more.
FLOOR = 2e-2


class _Base(torch.nn.Module):
    def __init__(self, dm):
        super().__init__()
        self.diffusion_model = dm


def tiny_model(seed=0):
    torch.manual_seed(seed)
    dm = mm_h3.MiniMaxH3Model(
        hidden_size=HIDDEN, num_layers=N_BLOCKS, token_refiner_num_layers=1,
        num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=384,
        text_dim=HIDDEN, timestep_input_dim=32, time_embed_hidden_size=64, time_embed_dim=64,
        dtype=torch.float32, device="cpu", operations=comfy.ops.manual_cast)
    with torch.no_grad():
        for name, p in dm.named_parameters():
            if "norm" in name:
                p.fill_(1.0)
            else:
                p.normal_(0.0, 0.05)
        n = dm.rope.inv_freq.numel()
        dm.rope.inv_freq.copy_(1.0 / (10000.0 ** (torch.arange(n, dtype=torch.float32) / n)))
    dm.requires_grad_(False)   # core's in-place rope refuses autograd
    base = _Base(dm)
    return comfy.model_patcher.ModelPatcher(base, load_device=torch.device("cpu"),
                                            offload_device=torch.device("cpu"))


def inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    video = torch.randn((1, 24) + LATENT, generator=g)
    audio = torch.randn((1, 32, 2, AUDIO_T), generator=g)
    context = torch.randn((1, TEXT_LEN, HIDDEN), generator=g).to(torch.bfloat16)
    return video, audio, context


def call(patcher, video, audio, context, sigma, video_mask: float | None = 0.0, audio_mask=None, extra=None):
    """One model call the way the sampler makes it: patches and wrappers in the options."""
    dm = patcher.get_model_object("diffusion_model")
    opts = comfy.model_patcher.copy_nested_dicts(
        patcher.model_options.get("transformer_options", {})) \
        if hasattr(comfy.model_patcher, "copy_nested_dicts") else \
        {k: (dict(v) if isinstance(v, dict) else v)
         for k, v in patcher.model_options.get("transformer_options", {}).items()}
    opts["wrappers"] = {k: dict(v) for k, v in patcher.wrappers.items()}
    opts["cond_or_uncond"] = [0]
    if extra:
        opts.update(extra)
    layout = mm_h3.PackedLayout(TEXT_LEN, *LATENT, AUDIO_T)
    kwargs = {"minimax_payload": {"layout": layout}}
    if video_mask is not None:
        kwargs["denoise_mask"] = torch.full((1, 1) + LATENT, float(video_mask))
    if audio_mask is not None:
        kwargs["audio_denoise_mask"] = torch.full((1, 1, 2, AUDIO_T), float(audio_mask))
    with torch.no_grad():
        out = dm.forward([video.clone(), audio.clone()], torch.tensor([sigma * 1000.0]),
                         context, transformer_options=opts, **kwargs)
    return [t.float() for t in out]


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


def main() -> int:
    problems = []

    def fail(msg):
        problems.append(msg)

    stock = tiny_model()
    video, audio, context = inputs()
    audio2 = audio + 0.5 * torch.randn_like(audio)

    def cached(precision="bf16", **kw):
        m = fvc.attach(stock, precision=precision, **kw)
        return m, m._h3_frozen_video_cache

    # 1. gate controls, each must run stock with every block counted stock
    m, st = cached()
    for label, vm, am in (("no mask", None, None), ("video mask 0.05", 0.05, None),
                          ("audio frozen", 0.0, 0.0)):
        call(m, video, audio, context, 0.5, video_mask=vm, audio_mask=am)
        rec = st.calls[-1]
        if rec["mode"] != "off" or rec["counts"]["stock"] != N_BLOCKS:
            fail(f"gate: {label} took {rec['mode']} with counts {rec['counts']}")

    # 2 and 3. build equals stock bit for bit; a cached step with nothing moved matches stock
    ref = call(stock, video, audio, context, 0.5)
    built = call(m, video, audio, context, 0.5)
    rec = st.calls[-1]
    if rec["mode"] != "build" or rec["counts"]["build"] != N_BLOCKS:
        fail(f"build call took {rec['mode']} with counts {rec['counts']}")
    if not torch.equal(built[1], ref[1]):
        fail(f"build audio differs from stock (rel {rel(built[1], ref[1]):.3g}); "
             "the build must be the stock forward")
    same = call(m, video, audio, context, 0.5)
    rec = st.calls[-1]
    if rec["mode"] != "cached" or rec["counts"]["cached"] != N_BLOCKS:
        fail(f"cached call took {rec['mode']} with counts {rec['counts']}")
    floor = rel(same[1], ref[1])
    if floor > FLOOR:
        fail(f"cached step with nothing moved is {floor:.3g} from stock (bound {FLOOR})")

    # 4. the severed edge is visible, and the live rows are live
    ref2 = call(stock, video, audio2, context, 0.3)
    moved = call(m, video, audio2, context, 0.3)
    err_moved = rel(moved[1], ref2[1])
    if err_moved <= 3 * max(floor, 1e-4):
        fail(f"moved audio: cached is {err_moved:.3g} from stock, not clearly above the "
             f"{floor:.3g} floor; the check cannot see the approximation")
    if not rel(moved[1], ref2[1]) < rel(moved[1], ref[1]):
        fail("moved audio: the cached step is closer to the build's stock output than to "
             "its own; the live rows are not live")

    # 5. text live beats text cached when the timestep moves
    ref3 = call(stock, video, audio, context, 0.3)
    m_live, _ = cached()
    call(m_live, video, audio, context, 0.5)
    live = call(m_live, video, audio, context, 0.3)
    saved = fvc.LIVE_KINDS
    try:
        fvc.LIVE_KINDS = ("audio",)
        m_dead, _ = cached()
        call(m_dead, video, audio, context, 0.5)
        dead = call(m_dead, video, audio, context, 0.3)
    finally:
        fvc.LIVE_KINDS = saved
    e_live, e_dead = rel(live[1], ref3[1]), rel(dead[1], ref3[1])
    if not e_live < e_dead:
        fail(f"text live ({e_live:.3g}) is not closer to stock than text cached ({e_dead:.3g})")

    # 6. rebuild on a moved pin, refusals, and the pass-scoped free
    call(m, video, audio, context, 0.0005)   # t_v 0.9995 passes the 0.999 video pin
    if st.calls[-1]["mode"] != "build" or "timestep" not in (st.calls[-1]["reason"] or ""):
        fail(f"a moved pin did not rebuild: {st.calls[-1]}")

    foreign = stock.clone()
    foreign.set_model_patch_replace(lambda a, e: e["original_block"](a), "dit", "double_block", 0)
    try:
        fvc.attach(foreign)
        fail("a model with a foreign dit replace was accepted")
    except ValueError:
        pass

    m_obj, st_obj = cached()
    blk0 = m_obj.get_model_object("diffusion_model").blocks[0]
    blk0.attn.forward = blk0.attn.forward        # an instance-level patch, as add_object_patch leaves
    try:
        call(m_obj, video, audio, context, 0.5)
        if st_obj.calls[-1]["reason"] != "foreign patch" or st_obj.calls[-1]["mode"] != "off":
            fail(f"an object-patched attention forward was not refused: {st_obj.calls[-1]}")
    finally:
        del blk0.attn.forward

    outer = m.wrappers[WrappersMP.OUTER_SAMPLE][fvc.WRAPPER_KEY][0]
    outer(lambda: None)
    if st.slots:
        fail("the OUTER_SAMPLE wrapper left the store allocated after the pass")

    # 7. codecs and verify
    t = torch.randn(300, 520) * 3.0
    for name, bound in (("bf16", 1e-2), ("fp8", 8e-2), ("int4", 0.2)):
        codec = fvc.CODECS[name]
        p, s = fvc._quantize_to_host(codec, t)
        out = fvc._dequantize_from_host(codec, p, s, torch.empty_like(t))
        if rel(out, t) > bound:
            fail(f"codec {name}: round trip {rel(out, t):.3g} over {bound}")
    m_v, st_v = cached(verify=True)
    call(m_v, video, audio, context, 0.5)
    call(m_v, video, audio2, context, 0.3)
    v = st_v.calls[-1].get("verify")
    if not v or not (0.0 < v["audio_cos"] <= 1.0 + 1e-6):
        fail(f"verify did not record a cosine: {st_v.calls[-1]}")

    print(f"  floor {floor:.3g}, moved audio {err_moved:.3g}, text live {e_live:.3g} "
          f"against text cached {e_dead:.3g}, verify cosine {v['audio_cos'] if v else None}")
    if problems:
        print(f"\n  FAIL  {len(problems)} problem(s):")
        for pr in problems:
            print(f"    - {pr}")
        return 1
    print("  ok    the gate sends stock what it should; the build is stock bit for bit; a "
          "cached step matches stock when nothing moved and visibly departs when the audio "
          "does; text live beats text cached; a moved pin rebuilds; foreign patches are "
          "refused; the pass frees the store; codecs and verify hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
