#!/usr/bin/env python3
"""`MiniMaxH3FrozenVideoCache` on a tiny H3 model: which path each call takes, and what it computes.

`frozen_video_cache.py` is the node. It changes what the model computes on a
cached step, so this runs core's real `MiniMaxH3Model` (three blocks, random
weights, bf16 compute on CPU) through the node's own patches and wrappers, the
way the sampler would, and grades:

1. **The path taken, by count.** A build call runs every block through the
   build path and a cached call runs every block cached. The gate sends a
   call with no mask, a video open everywhere (0.05) with nothing frozen, or
   a call that regenerates nothing, stock. A check that only compared outputs
   would pass vacuously on a call that never reached the cache.
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
5. **Live rows are live from the first block.** A kept row's input to block
   0 is the same on every step, so block 0 of a cached step is exact on
   every live row whatever moved. With the timestep moved, the text and
   audio rows after block 0 must be stock's, and with text cached
   (`ALWAYS_LIVE_KINDS` monkeypatched to nothing) the text rows must be far
   from it, on each of `SEEDS`. Until 2026-10-06 this item asserted that
   text live left the audio closer to stock than text cached; on this model
   that margin is at bf16's resolution and held by the seed.
6. **Rebuild and refusal.** A timestep past the cond pin rebuilds. A foreign
   `patches_replace["dit"]` entry is refused when the node attaches, and an
   object-patched attention forward sends the call stock with a reason. The
   OUTER_SAMPLE wrapper frees the store when the pass ends.
7. **The codecs round-trip within their bounds**, and `verify` records the
   cosine it promises.
8. **`verify` times the cached step by stage, and only `verify` does.** A
   verified cached call records every name in `STAGES`, and they add up to
   the call's own seconds; a build, and a cached call with `verify` off,
   record none. The control: a stage's lap removed from the block must leave
   that stage at zero, so the check would see a stage that was never timed.

Items 9 to 16 are the masked window: one region of the video regenerated,
every other video row kept. Its layout also carries a reference image's rows
and a text span cut into tag runs, which the refine items do not have, and
its token grid has three different sides, so height and width cannot be
swapped unseen.

9. **The live rows are the ones core regenerates, by name, and each runs at
   its own timestep.** The set the module computes is compared with one built
   here from the token grid, and the build is the stock forward bit for bit
   on the video. Every table of modulation rows the module hands the cached
   block is compared, row by row, with core's own for the same packed rows;
   at least one of those tables must be core's per-row kind, or the case
   never arose. Outputs alone do not hold this: on this model a region run
   at the kept rows' timestep stays inside item 10's bound (seen 2026-10-06,
   by breaking the gather on purpose).
10. **A cached step with nothing moved matches stock on the regenerated
    rows**, and the stock output there is not all zero, so the comparison is
    not empty.
11. **The control.** Move the latent under the mask and the timestep: the
    cached step must depart from stock clearly above item 10's floor and
    stay closer to its own stock step than to the build's.
12. **A wrong live set is visible.** Item 10 cannot see one: with nothing
    moved every kept row is exactly what stock would compute, whichever rows
    are recomputed. So the moved step is run again with the video's live
    rows shifted by one, and it must be further from stock than the right
    set is.
13. **The gate's other answers.** A different region rebuilds, with the
    reason. A region too large to gain runs stock, with the reason and the
    share on the record. A mask on another grid than the layout's, in any
    axis, is turned away without raising. A context window runs stock, with
    the reason. An audio mask open on some rows and frozen on others runs
    cached on the open ones, by name, and matches stock when nothing moved.
14. **The kept content is the build's.** Same region, same conditioning
    index, but the latent under the kept tokens, or the reference latent, is
    another one: the call must rebuild, with the reason. Without the guard
    it would be served the first content's rows.
15. **The halo.** One token of halo makes live exactly the kept rows within
    a token of the region, compared by name with a set built here cell by
    cell, on a token at the grid's edge. With every row live (a halo across
    the whole grid, audio open, no reference, the share limit lifted for
    this one call) the moved step matches stock: nothing is kept, so nothing
    is stale, and what is left is the cached block's own arithmetic on a
    per-row timestep table.
16. **`verify` on a masked window** records the video's cosine and relative
    L2, counts the ring and the interior as a count made here cell by cell
    does, runs the step again with the text cached and gets a different
    answer, and returns the same output a plain cached step does.

**Not graded: whether a halo, or live text, brings a masked step closer to
stock.** On this model the kept rows' reaction to the region is at bf16's
resolution: over four seeds and four moves a one-token halo came out closer
to stock about as often as further (run 2026-10-06, a one-off probe). An
ordering asserted here would hold by the seed. It is `verify`'s to measure
on a render.

**Assertions here that cannot fail today**, kept and marked where they
stand: a build recording stages (timing starts only on a cached call).

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

import hashlib  # noqa: E402

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
LATENT = (3, 8, 12)      # latent t, h, w: a token grid of 3 x 4 x 6, three different sides, and
                         # enough video that the refine pass's live rows (text and audio) stay
                         # under `LIVE_SHARE_LIMIT`
GRID = (LATENT[0], LATENT[1] // 2, LATENT[2] // 2)
REF_LATENT = (1, 4, 4)   # the masked items' reference image: four packed rows
TEXT_TAGS = (1, 1, 0, 0, 1, 1, 1, 0)    # the masked items' text span, in four tag runs
SEEDS = (0, 10, 20)      # item 5's models
AUDIO_T = 10
#: Item 3's bound. **Reasoned**: bf16 carries 8 mantissa bits, so a matching
#: forward agrees to a few 1e-3 in relative L2; a severed edge moves it more.
FLOOR = 2e-2
#: Item 5's two bounds on rows after block 0, relative L2 against stock.
#: **Measured** on this model over `SEEDS`, 2026-10-06: live rows are exactly
#: stock's on every seed, and cached text, left at its embedding, is never
#: under 2e-2 away (one block moves a row that little here). **Reasoned**: the
#: far bound is a third of the smallest measured, the near one far above any
#: rounding and a sixth of the far one.
BLOCK0_SAME, BLOCK0_FAR = 1e-3, 6e-3


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


def packed_layout(rich=False):
    """The refine items' layout, or with `rich` the masked items': a reference image's rows as well."""
    refs = [{"kind": "image", "latent_h": REF_LATENT[1], "latent_w": REF_LATENT[2]}] if rich else None
    return mm_h3.PackedLayout(TEXT_LEN, *LATENT, AUDIO_T, refs=refs)


def reference(seed=5):
    return torch.randn((1, 24) + REF_LATENT, generator=torch.Generator().manual_seed(seed))


def call(patcher, video, audio, context, sigma, video_mask: float | None = 0.0, audio_mask=None, extra=None,
         rich=False, ref=None):
    """One model call the way the sampler makes it: patches and wrappers in the options.

    `rich` adds what a masked window's call carries and the refine pass's does
    not: a reference image's rows (`ref`, or `reference()`) and text tags.
    """
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
    payload = {"layout": packed_layout(rich)}
    if rich:
        z = reference() if ref is None else ref
        payload.update(refs=[{"kind": "image", "latent_h": REF_LATENT[1], "latent_w": REF_LATENT[2], "latent": z}],
                       cond_video_latents=[z], text_token_tags=torch.tensor([TEXT_TAGS]))
    kwargs = {"minimax_payload": payload}
    if isinstance(video_mask, torch.Tensor):
        kwargs["denoise_mask"] = video_mask.clone()
    elif video_mask is not None:
        kwargs["denoise_mask"] = torch.full((1, 1) + LATENT, float(video_mask))
    if isinstance(audio_mask, torch.Tensor):
        kwargs["audio_denoise_mask"] = audio_mask.clone()
    elif audio_mask is not None:
        kwargs["audio_denoise_mask"] = torch.full((1, 1, 2, AUDIO_T), float(audio_mask))
    with torch.no_grad():
        out = dm.forward([video.clone(), audio.clone()], torch.tensor([sigma * 1000.0]),
                         context, transformer_options=opts, **kwargs)
    return [t.float() for t in out]


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


def after_block0(patcher, fn):
    """Run `fn()` and return the packed rows as block 0 left them, [rows, hidden], float.

    A stock call runs the block module, which a forward hook sees; a cached
    call never does, so the module's own cached block is watched as well.
    """
    seen = []
    blk = patcher.get_model_object("diffusion_model").blocks[0]
    hook = blk.register_forward_hook(lambda m, a, out: seen.append(out.float().clone()))
    saved = fvc._cached_block

    def spy(state, i, args):
        r = saved(state, i, args)
        if i == 0:
            seen.append(r["img"].float().clone())
        return r

    fvc._cached_block = spy
    try:
        fn()
    finally:
        fvc._cached_block = saved
        hook.remove()
    return seen[-1]


def region(t, h, w):
    """A latent denoise mask and its token grid: 1 over the token cells in the three ranges, 0 elsewhere."""
    grid = torch.zeros(GRID, dtype=torch.bool)
    grid[t[0]:t[1], h[0]:h[1], w[0]:w[1]] = True
    mask = grid.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1).to(torch.float32)[None, None]
    return mask, grid


def near_kept(grid):
    """The cells of `grid` with a cell outside it within one token, counted one cell at a time."""
    out = torch.zeros(GRID, dtype=torch.bool)
    for t, h, w in grid.nonzero().tolist():
        for dt in (-1, 0, 1):
            for dh in (-1, 0, 1):
                for dw in (-1, 0, 1):
                    c = (t + dt, h + dh, w + dw)
                    if all(0 <= c[k] < GRID[k] for k in range(3)) and not bool(grid[c]):
                        out[t, h, w] = True
    return out


def within_one(grid):
    """`grid` and every in-grid cell within one token of it, one cell at a time."""
    out = grid.clone()
    for t, h, w in grid.nonzero().tolist():
        for dt in (-1, 0, 1):
            for dh in (-1, 0, 1):
                for dw in (-1, 0, 1):
                    c = (t + dt, h + dh, w + dw)
                    if all(0 <= c[k] < GRID[k] for k in range(3)):
                        out[c] = True
    return out


def live_sig(grid, audio_live=None, rich=True):
    """The digest and count of the live rows this check expects, built from the layout and nothing else.

    `grid` is the token grid of live video rows; `audio_live` a bool per
    audio row, or None for audio frozen whole.
    """
    layout = packed_layout(rich)
    live = torch.zeros(layout.seq_len, dtype=torch.bool)
    for a, b, kind in layout.segments:
        if kind == "text":
            live[a:b] = True
        elif kind == "audio" and audio_live is not None:
            live[a:b] = audio_live
        elif kind == "video":
            live[a:b] = grid.reshape(-1)
    return hashlib.blake2b(live.numpy().tobytes(), digest_size=8).hexdigest(), int(live.sum())


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
    for label, vm, am, why in (("no mask", None, None, "no mask"),
                               ("video mask 0.05", 0.05, None, "nothing frozen"),
                               ("audio frozen", 0.0, 0.0, "nothing regenerates")):
        call(m, video, audio, context, 0.5, video_mask=vm, audio_mask=am)
        rec = st.calls[-1]
        if rec["mode"] != "off" or rec["counts"]["stock"] != N_BLOCKS or rec["reason"] != why:
            fail(f"gate: {label} took {rec['mode']} with counts {rec['counts']} and reason {rec['reason']!r}")

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

    # 5. live rows are stock's after block 0 whatever moved; cached text is not
    layout0 = packed_layout()
    text_rows = torch.cat([torch.arange(a, b) for a, b, k in layout0.segments if k == "text"])
    live_rows = torch.cat([torch.arange(a, b) for a, b, k in layout0.segments if k in ("text", "audio")])
    block0 = []
    for seed in SEEDS:
        model = tiny_model(seed)
        v_s, a_s, c_s = inputs(seed + 1)
        a_s2 = a_s + 0.5 * torch.randn_like(a_s)
        want0 = after_block0(model, lambda: call(model, v_s, a_s2, c_s, 0.3))
        m_live = fvc.attach(model, precision="bf16")
        call(m_live, v_s, a_s, c_s, 0.5)
        got0 = after_block0(m_live, lambda: call(m_live, v_s, a_s2, c_s, 0.3))
        saved = fvc.ALWAYS_LIVE_KINDS
        try:
            fvc.ALWAYS_LIVE_KINDS = ()
            m_dead = fvc.attach(model, precision="bf16")
            call(m_dead, v_s, a_s, c_s, 0.5)
            dead0 = after_block0(m_dead, lambda: call(m_dead, v_s, a_s2, c_s, 0.3))
        finally:
            fvc.ALWAYS_LIVE_KINDS = saved
        modes = (m_live._h3_frozen_video_cache.calls[-1]["mode"], m_dead._h3_frozen_video_cache.calls[-1]["mode"])
        if modes != ("cached", "cached"):
            fail(f"item 5, seed {seed}: the calls took {modes}, not cached")
        block0.append((rel(got0[live_rows], want0[live_rows]), rel(dead0[text_rows], want0[text_rows])))
    e_live, e_dead = max(a for a, _ in block0), min(b for _, b in block0)
    if e_live > BLOCK0_SAME:
        fail(f"live rows after block 0 are up to {e_live:.3g} from stock over seeds {SEEDS} (bound {BLOCK0_SAME})")
    if e_dead < BLOCK0_FAR:
        fail(f"text cached: the text rows after block 0 are as little as {e_dead:.3g} from stock over seeds "
             f"{SEEDS} (at least {BLOCK0_FAR} expected); the control does not cache the text")

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

    # 8. the stage split: there under verify on a cached call, adding up, and nowhere else
    built_rec, cached_rec = st_v.calls[-2], st_v.calls[-1]
    stages = cached_rec.get("stages")
    # Guards nothing today: timing starts only on a cached call, so a build cannot
    # reach the line that records stages. Kept for the day a build is timed too.
    if "stages" in built_rec:
        fail(f"a build call recorded stages: {built_rec['stages']}")
    if not stages or tuple(stages) != fvc.STAGES:
        fail(f"a verified cached call did not record {fvc.STAGES}: {stages}")
    else:
        total = sum(stages.values())
        if min(stages.values()) <= 0.0:
            fail(f"a stage was never timed: {stages}")
        if not 0.9 * cached_rec["seconds"] <= total <= cached_rec["seconds"] + 1e-3:
            fail(f"the stages add up to {total:.4f} s and the call took {cached_rec['seconds']:.4f} s")
    if any("stages" in r for r in st.calls):
        fail("a call with verify off recorded stages, so it waited for the card")
    saved_lap = fvc._lap
    try:
        fvc._lap = lambda state, name, device: None if name == "attention" else saved_lap(state, name, device)
        m_c, st_c = cached(verify=True)
        call(m_c, video, audio, context, 0.5)
        call(m_c, video, audio2, context, 0.3)
        if st_c.calls[-1].get("stages", {}).get("attention", 1.0) != 0.0:
            fail(f"control: with the attention lap removed the stage still reads "
                 f"{st_c.calls[-1].get('stages')}; item 8 cannot see an untimed stage")
    finally:
        fvc._lap = saved_lap

    # ---- the masked window: one video region regenerated, a reference image and tagged text in the layout ----
    mask, grid = region((1, 3), (1, 3), (0, 2))             # 8 of the 72 video rows; latent step 0 kept whole
    sig, n_live = live_sig(grid)
    video2 = video + 0.5 * torch.randn_like(video) * mask[0]   # the latent moves under the mask only
    window = dict(video_mask=mask, audio_mask=0.0, rich=True)

    # 9. the live rows by name, each at its own timestep, and the build is stock
    saved_segments = fvc._live_segments
    audits = []

    def audited(mod_segments, idx):
        got = saved_segments(mod_segments, idx)
        want = []
        for i in idx.tolist():
            a, _b, row = next(seg for seg in mod_segments if seg[0] <= i < seg[1])
            want.append(int(row[i - a]) if isinstance(row, torch.Tensor) else int(row))
        flat = []
        for lo, hi, r in got or []:
            flat += [int(v) for v in r.tolist()] if isinstance(r, torch.Tensor) else [int(r)] * (hi - lo)
        audits.append((flat == want, any(isinstance(seg[2], torch.Tensor) for seg in mod_segments), len(mod_segments)))
        return got

    fvc._live_segments = audited
    mm, stm = cached()
    refm = call(stock, video, audio, context, 0.5, **window)
    builtm = call(mm, video, audio, context, 0.5, **window)
    rec = stm.calls[-1]
    if rec["mode"] != "build" or rec["counts"]["build"] != N_BLOCKS:
        fail(f"masked build took {rec['mode']} with counts {rec['counts']} ({rec['reason']})")
    if rec.get("live_sig") != sig or rec.get("live_rows") != n_live:
        fail(f"masked window: the module's live rows ({rec.get('live_rows')}, {rec.get('live_sig')}) "
             f"are not the regenerated tokens and the text ({n_live}, {sig})")
    if not torch.equal(builtm[0], refm[0]):
        fail(f"masked build video differs from stock (rel {rel(builtm[0], refm[0]):.3g})")

    # 10. nothing moved: the regenerated rows match stock, and there is something to compare
    samem = call(mm, video, audio, context, 0.5, **window)
    rec = stm.calls[-1]
    if rec["mode"] != "cached" or rec["counts"]["cached"] != N_BLOCKS:
        fail(f"masked cached call took {rec['mode']} with counts {rec['counts']} ({rec['reason']})")
    floor_m = rel(samem[0], refm[0])
    if floor_m > FLOOR:
        fail(f"masked cached step with nothing moved is {floor_m:.3g} from stock (bound {FLOOR})")
    if float(refm[0].abs().max()) == 0.0:
        fail("masked window: the stock video output is all zero; the comparison is empty")

    # 11. the control: the latent under the mask and the timestep move
    refm2 = call(stock, video2, audio, context, 0.3, **window)
    movedm = call(mm, video2, audio, context, 0.3, **window)
    if stm.calls[-1]["mode"] != "cached":
        fail(f"masked moved call took {stm.calls[-1]['mode']} ({stm.calls[-1]['reason']})")
    e_right = rel(movedm[0], refm2[0])
    if e_right <= 3 * max(floor_m, 1e-4):
        fail(f"masked window, moved: cached is {e_right:.3g} from stock, not clearly above the "
             f"{floor_m:.3g} floor; the check cannot see the approximation")
    if not e_right < rel(movedm[0], refm[0]):
        fail("masked window, moved: the cached step is closer to the build's stock output than "
             "to its own; the regenerated rows are not live")

    fvc._live_segments = saved_segments
    if not audits or not all(ok for ok, _, _ in audits):
        fail(f"masked window: {sum(not ok for ok, _, _ in audits)} of {len(audits)} tables of modulation rows "
             "for the live rows are not core's rows for the same packed rows")
    if not any(per_row for _, per_row, _ in audits):
        fail("masked window: core never handed a per-row timestep table, so item 9 audited the easy case only")
    if not any(n >= len(set(TEXT_TAGS)) + 5 for _, _, n in audits):
        fail("masked window: the text span was never cut into tag runs, so item 9 audited one text segment only")

    # 12. a live set shifted by one row is further from stock than the right one
    saved_rows = fvc._rows

    def shifted(layout, video_mask, audio_mask, halo=0):
        r = saved_rows(layout, video_mask, audio_mask, halo)
        live = r.live.clone()
        va, vb = next((a, b) for a, b, kind in layout.segments if kind == "video")
        live[va:vb] = torch.roll(r.live[va:vb], 1)
        return fvc._Rows(live, r.regen, r.audio, r.text, 0)

    try:
        fvc._rows = shifted
        ms, sts = cached()
        call(ms, video, audio, context, 0.5, **window)
        wrong = call(ms, video2, audio, context, 0.3, **window)
        if sts.calls[-1]["mode"] != "cached" or sts.calls[-1].get("live_sig") == sig:
            fail(f"control: the shifted live set did not run cached on other rows: {sts.calls[-1]}")
    finally:
        fvc._rows = saved_rows
    e_wrong = rel(wrong[0], refm2[0])
    if not e_right < e_wrong:
        fail(f"masked window: the right live rows ({e_right:.3g}) are not closer to stock than "
             f"rows shifted by one ({e_wrong:.3g}); a wrong live set would pass")

    # 13. the gate's other answers
    mask_b, _grid_b = region((0, 2), (2, 4), (2, 4))
    call(mm, video, audio, context, 0.5, video_mask=mask_b, audio_mask=0.0, rich=True)
    if stm.calls[-1]["mode"] != "build" or stm.calls[-1]["reason"] != "the frozen rows changed":
        fail(f"a different region did not rebuild: {stm.calls[-1]}")
    mask_big, _g = region((0, 3), (0, 4), (0, 4))           # 48 of the 72 video rows
    call(mm, video, audio, context, 0.5, video_mask=mask_big, audio_mask=0.0, rich=True)
    rec = stm.calls[-1]
    if rec["mode"] != "off" or rec["counts"]["stock"] != N_BLOCKS \
            or rec["reason"] != "live share above the limit" or not rec["live_share"] > fvc.LIVE_SHARE_LIMIT:
        fail(f"a region over the share limit did not run stock with the reason: {rec}")

    gate_kwargs = {"minimax_payload": {"layout": packed_layout(True)},
                   "audio_denoise_mask": torch.zeros((1, 1, 2, AUDIO_T))}
    took = fvc._gate([video, audio], dict(gate_kwargs, denoise_mask=mask), {}, 0)
    if took[0] is None:
        fail(f"the gate did not take the window's own mask ({took[2]}); the misfit cases below prove nothing")
    t, h, w = LATENT
    for label, shape in (("a shorter mask", (t - 1, h, w)), ("a longer mask", (t + 1, h, w)),
                         ("a lower mask", (t, h - 2, w)), ("a wider mask", (t, h, w + 2)),
                         ("a mask one row short, an odd latent", (t, h - 1, w))):
        bad = torch.zeros((1, 1) + shape)
        bad[:, :, :1] = 1.0
        try:
            got = fvc._gate([video, audio], dict(gate_kwargs, denoise_mask=bad), {}, 0)
        except Exception as exc:            # noqa: BLE001 - the point is that it must not raise
            fail(f"{label} made the gate raise {type(exc).__name__}: {exc}")
            continue
        if got[0] is not None or got[2] != "the masks do not fit the layout":
            fail(f"{label} was not turned away: {got[2]!r}")

    mw, stw = cached()
    call(mw, video, audio, context, 0.5, extra={"context_window": object()}, **window)
    rec = stw.calls[-1]
    if rec["mode"] != "off" or rec["counts"]["stock"] != N_BLOCKS or rec["reason"] != "a context window":
        fail(f"a call under a context window did not run stock with the reason: {rec}")

    amask = torch.zeros((1, 1, 2, AUDIO_T))
    amask[..., : AUDIO_T // 2] = 1.0                        # half of each audio channel's rows open
    part = dict(video_mask=mask, audio_mask=amask, rich=True)
    ma, sta = cached()
    refa = call(stock, video, audio, context, 0.5, **part)
    call(ma, video, audio, context, 0.5, **part)
    samea = call(ma, video, audio, context, 0.5, **part)
    rec = sta.calls[-1]
    if rec["mode"] != "cached" or rec.get("live_sig") != live_sig(grid, amask[0, 0].reshape(-1) > 0.5)[0]:
        fail(f"a partly open audio mask did not run cached on text, the open audio rows and the region: {rec}")
    if max(rel(samea[0], refa[0]), rel(samea[1], refa[1])) > FLOOR or float(refa[1].abs().max()) == 0.0:
        fail(f"a partly open audio mask, nothing moved: video {rel(samea[0], refa[0]):.3g}, "
             f"audio {rel(samea[1], refa[1]):.3g} from stock (bound {FLOOR}), or the stock audio is all zero")

    # 14. the kept content is the build's: another latent under the kept tokens, or another reference, rebuilds
    mc, stc = cached()
    call(mc, video, audio, context, 0.5, **window)
    call(mc, video, audio, context, 0.5, **window)
    if stc.calls[-1]["mode"] != "cached":
        fail(f"item 14's second call was not cached: {stc.calls[-1]}")
    other = video + 0.5 * torch.randn_like(video) * (1.0 - mask[0])   # another plate, the same region
    call(mc, other, audio, context, 0.5, **window)
    if stc.calls[-1]["mode"] != "build" or stc.calls[-1]["reason"] != "the kept rows' input changed":
        fail(f"another latent under the kept tokens did not rebuild: {stc.calls[-1]}")
    call(mc, other, audio, context, 0.5, **window)
    call(mc, other, audio, context, 0.5, ref=reference(seed=6), **window)
    if stc.calls[-1]["mode"] != "build" or stc.calls[-1]["reason"] != "the kept rows' input changed":
        fail(f"another reference latent did not rebuild: {stc.calls[-1]}")
    saved_same = fvc._same_content
    try:
        fvc._same_content = lambda a, b: True
        mg, stg = cached()
        call(mg, video, audio, context, 0.5, **window)
        call(mg, other, audio, context, 0.5, **window)
        if stg.calls[-1]["mode"] != "cached":
            fail(f"control: with the content comparison forced equal the call still took {stg.calls[-1]['mode']} "
                 f"({stg.calls[-1]['reason']}); something else rebuilt it, so item 14 does not hold the guard")
    finally:
        fvc._same_content = saved_same

    # 15. the halo: the rows it adds, by name, against a set built cell by cell
    mask_1, grid_1 = region((1, 2), (0, 1), (1, 2))         # one token on the grid's edge
    near = within_one(grid_1)
    video3 = video + 0.5 * torch.randn_like(video) * mask_1[0]
    for halo, want_grid in ((0, grid_1), (1, near)):
        mh, sth = cached(halo=halo)
        call(mh, video, audio, context, 0.5, video_mask=mask_1, audio_mask=0.0, rich=True)
        call(mh, video3, audio, context, 0.3, video_mask=mask_1, audio_mask=0.0, rich=True)
        rec = sth.calls[-1]
        want_sig, want_live = live_sig(want_grid)
        added = int(want_grid.sum()) - int(grid_1.sum())
        if rec["mode"] != "cached" or rec.get("live_sig") != want_sig or rec["live_rows"] != want_live \
                or rec["halo_rows"] != added:
            fail(f"halo {halo}: {rec['mode']}, {rec.get('live_rows')} live rows of which "
                 f"{rec.get('halo_rows')} halo, expected {want_live} of which {added}, on the rows named")
    if int(near.sum()) in (int(grid_1.sum()), 27):
        fail("halo: the edge token's neighbourhood is not clipped by the grid; the case is the easy one")
    saved_limit = fvc.LIVE_SHARE_LIMIT
    try:
        fvc.LIVE_SHARE_LIMIT = 1.0
        mall, stall = cached(halo=max(GRID))
        refall = call(stock, video2, audio2, context, 0.3, video_mask=mask)
        call(mall, video, audio, context, 0.5, video_mask=mask)
        gotall = call(mall, video2, audio2, context, 0.3, video_mask=mask)
        rec = stall.calls[-1]
    finally:
        fvc.LIVE_SHARE_LIMIT = saved_limit
    e_all = max(rel(gotall[0], refall[0]), rel(gotall[1], refall[1]))
    if rec["mode"] != "cached" or rec["live_rows"] != rec["rows"]:
        fail(f"every row live: took {rec['mode']} with {rec.get('live_rows')} of {rec.get('rows')} rows")
    elif e_all > FLOOR:
        fail(f"every row live and everything moved: {e_all:.3g} from stock (bound {FLOOR}); "
             "the cached block's own arithmetic differs from the stock block's")

    # 16. verify on a masked window
    mask_c, grid_c = region((0, 3), (1, 4), (1, 5))         # 36 rows, a third of them with no kept neighbour
    ring_c = near_kept(grid_c)
    want_ring, want_inner = int(ring_c.sum()), int(grid_c.sum()) - int(ring_c.sum())
    video4 = video + 0.5 * torch.randn_like(video) * mask_c[0]
    mv, stv = cached(verify=True)
    call(mv, video, audio, context, 0.5, video_mask=mask_c, audio_mask=0.0, rich=True)
    out_v = call(mv, video4, audio, context, 0.3, video_mask=mask_c, audio_mask=0.0, rich=True)
    vm = stv.calls[-1].get("verify") or {}
    mp, _stp = cached()
    call(mp, video, audio, context, 0.5, video_mask=mask_c, audio_mask=0.0, rich=True)
    out_p = call(mp, video4, audio, context, 0.3, video_mask=mask_c, audio_mask=0.0, rich=True)
    if not torch.equal(out_v[0], out_p[0]):
        fail("verify changed the output of a cached step on a masked window")
    if not want_ring or not want_inner:
        fail(f"item 16's region has {want_ring} ring and {want_inner} interior rows; it needs both")
    if not (0.0 < vm.get("video_cos", 0.0) <= 1.0 + 1e-6) or "audio_cos" in vm:
        fail(f"verify on a masked window did not record the video alone: {vm}")
    elif (vm.get("ring_rows"), vm.get("interior_rows")) != (want_ring, want_inner):
        fail(f"verify's ring and interior are {vm.get('ring_rows')} and {vm.get('interior_rows')} rows; "
             f"counted cell by cell they are {want_ring} and {want_inner}")
    elif vm.get("text_cached_video_rel_l2") in (None, vm["video_rel_l2"]):
        fail(f"verify: the step run again with the text cached gave {vm.get('text_cached_video_rel_l2')} "
             f"against {vm['video_rel_l2']} with it live; the second pass did not cache the text")

    print(f"  masked window: floor {floor_m:.3g}, moved {e_right:.3g}, rows shifted by one {e_wrong:.3g}, "
          f"every row live {e_all:.3g}, verify ring {vm.get('ring_rows')} rows and interior "
          f"{vm.get('interior_rows')}")
    print(f"  floor {floor:.3g}, moved audio {err_moved:.3g}, live rows after block 0 at most {e_live:.3g} and "
          f"cached text at least {e_dead:.3g} from stock over seeds {SEEDS}, "
          f"verify cosine {v['audio_cos'] if v else None}")
    if problems:
        print(f"\n  FAIL  {len(problems)} problem(s):")
        for pr in problems:
            print(f"    - {pr}")
        return 1
    print("  ok    the gate sends stock what it should; the build is stock bit for bit; a "
          "cached step matches stock when nothing moved and visibly departs when the audio "
          "does; live rows are stock's after block 0 and cached text is not; a moved pin rebuilds; "
          "foreign patches are refused; the pass frees the store; codecs and verify hold; verify "
          "alone times the cached step, by stage; on a masked window the live rows are the "
          "regenerated tokens and the text, each at core's timestep row, a cached step matches "
          "stock when nothing moved, departs when the region does and more so on the wrong rows, "
          "a new region or new kept content rebuilds, a large region, a misfit mask and a context "
          "window run stock, a halo adds the rows named and a full one is exact, and verify "
          "grades the video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
