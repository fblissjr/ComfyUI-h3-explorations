"""Cache the frozen rows of an H3 sampling run, so its steps run on the live rows only.

Two passes freeze most of what they run. The audio-refine pass
(`audio_refine.py`, `h3_config.AUDIO_REFINE`) keeps the video exactly and
reopens the audio. A masked video-to-video window (`video_mask.py`,
`audio_freeze_song.py`) keeps the audio, the previous window's tail and every
video token outside the subject, and regenerates the rest. Either way every
step still runs the whole packed sequence. A row at mask 0 sees the same input
on every step (`comfy/model_base.py::MiniMaxH3.scale_latent_inpaint` injects
the same blend each call) at the same pinned timestep
(`comfy/ldm/minimax/model.py::MiniMaxH3Model._forward`, `t_pin_v` and
`t_pin_a`). What still moves between steps is the rows being generated and the
text: text rows follow the video stream's timestep, which falls every step.

So the first model call of the run (the build) runs every block stock and
keeps each block's attention input, the post-norm modulated hidden state. Each
later call (a cached step) computes only the live rows, and their queries
attend against K/V rebuilt from the kept hidden state with the live rows
written in. The cached rows stop reacting to the live ones: that is the
approximation, and the `verify` switch measures it on a render. `verify` also
splits each cached step's time into `STAGES`, which is the only place the step
waits for the card between stages.

**Which rows are live** is read per row from the masks core hands the model
(`_rows`): text always, a video row where core's own pooling
(`mask_row_values`) leaves its mask above `FROZEN_BELOW`, an audio row the
same. Conditioning and reference rows are cached. `halo` adds the kept video
rows within that many tokens of a regenerated one: they are recomputed each
cached step and never regenerated, so the subject is drawn against
neighbours that have seen it. A call is left stock, with the reason on its
record, when nothing is frozen, when nothing regenerates, or when the live
rows are more of the sequence than `LIVE_SHARE_LIMIT`.

**Ported from** Adudeguyman's ComfyUI-H3-AudioRefine (`frozen_cache.py`,
`coderef/ComfyUI-H3-AudioRefine`), MIT, notice below. The codecs are theirs
nearly verbatim; the rest is rewritten against current core. What differs:

- **The build is the stock block.** It runs through `original_block` with a
  capturing `attention=` callable, so the build step is the stock forward by
  construction, on whatever attention the graph wires (Sol, the kitchen
  backend). AudioRefine re-implements the block and calls attention without
  core's `preferred_attention`.
- **Text rows are live.** AudioRefine's docstring says text rows see a
  constant timestep; core gives them the video stream's `t_v`.
- **A mixed mask.** AudioRefine caches a wholly frozen video. Here the live
  rows are an index set, so a window that regenerates one region runs too.
- **One sampling run, never longer.** The store is freed when the pass ends.
  AudioRefine keeps it across prompts behind a sum/abs-sum fingerprint, which
  nothing shows can tell two seeds of one prompt apart.
- **Refuses what it would silently skip**: a foreign `patches_replace["dit"]`
  entry (core VSA, FunControl) at patch time, and an object-patched
  block or attention forward (sage capture, `exact_blocks.py`) at run time.
- **RAM store and the hidden contents only.** No VRAM or disk backend and no
  K/V contents.

Copyright (c) 2026 Adudeguyman, for the ported portions:

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque

import torch
from comfy_api.latest import io

import comfy.model_management
import comfy.model_prefetch
import comfy.quant_ops
import comfy.ldm.minimax.model as mm_h3
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
from comfy.patcher_extension import WrappersMP

log = logging.getLogger(__name__)

WRAPPER_KEY = "h3_frozen_video_cache"

#: Row kinds recomputed on every cached step whatever the masks say.
#: **Reasoned** from `MiniMaxH3Model._forward`'s `seg_t`: text follows `t_v`,
#: which falls every step. The check's item 5 is the control: text cached is
#: further from stock than text live.
ALWAYS_LIVE_KINDS = ("text",)

#: Row kinds whose rows are live or kept one by one, from the mask core hands
#: the model. Every other kind (cond, ref_img, cond_audio, ref_audio) is
#: cached: **reasoned** from `seg_t`, which pins them at the cond timestep.
MASKED_KINDS = ("video", "audio")

#: A row whose pooled mask is under this is frozen. **Inherited** from
#: AudioRefine's gate. Core rounds a mask up to a 256th
#: (`MiniMaxH3._token_grid_masks`), so under this means exactly 0.
FROZEN_BELOW = 1e-3

#: A call whose live rows are more of the packed sequence than this runs
#: stock. A cached step pays for every row once (the store to the card, qkv,
#: the attention call's own cost) and then for a dense rectangle of live
#: queries against all keys, where the stock step with Sol-Attn pays for a
#: sparse square. **Measured**: the first part, on the refine pass
#: (`bench/results/2026-10-06_frozen_cache_stage_split.md`), and the whole
#: step on one masked window
#: (`bench/results/2026-10-06_frozen_cache_masked_window.md`). **This limit
#: does not mark break-even.** The model that put break-even a little above
#: it assumed the rectangle's time follows the number of query rows; on the
#: masked window, at a live share well under this limit, the cached step is
#: slower than a Sol-Attn stock step and faster only than a dense one. So on
#: a Sol-Attn graph no share under this limit is known to gain. The limit is
#: **kept as inherited from that model**, as the bound above which the cache
#: does not try at all; where a cached step would have to get cheaper for a
#: masked window to gain is in the second record, not in this number.
LIVE_SHARE_LIMIT = 0.5

#: How many cells of each kept input `_content_sample` keeps. **Reasoned**: a
#: different clip, window or reference differs nearly everywhere, so a few
#: thousand cells compared exactly tell two apart, and the gather is small
#: beside a step.
CONTENT_SAMPLE = 4096

#: `verify`'s ring: regenerated video rows within this many tokens of a kept
#: one, in time, height or width. **Reasoned**: the rows drawn against stale
#: neighbours first.
RING_TOKENS = 1

#: int4 group size. **Inherited** from AudioRefine.
INT4_GROUP = 128

#: Rows per quantize/dequantize chunk, to bound the fp32 transients. All codecs
#: are row-independent, so chunking is exact. **Inherited** from AudioRefine.
QUANT_CHUNK_ROWS = 16384

_FP8 = getattr(torch, "float8_e4m3fn", None)


# ---------------------------------------------------------------------------
# codecs (ported from AudioRefine)

class _CodecBF16:
    name = "bf16"

    def quantize(self, t):
        return t.to(torch.bfloat16).contiguous(), None

    def dequantize(self, payload, scales, out):
        out.copy_(payload)
        return out

    def payload_shape(self, s, d):
        return (s, d), torch.bfloat16

    def scales_shape(self, s, d):
        return None, None


class _CodecFP8:
    name = "fp8"

    def quantize(self, t):
        scales = t.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-8) / 448.0
        return (t / scales).clamp(-448.0, 448.0).to(_FP8).contiguous(), scales.contiguous()

    def dequantize(self, payload, scales, out):
        tmp = payload.to(torch.float32)
        tmp.mul_(scales)
        out.copy_(tmp)
        return out

    def payload_shape(self, s, d):
        return (s, d), _FP8

    def scales_shape(self, s, d):
        return (s, 1), torch.float32


class _CodecINT4:
    name = "int4"

    def quantize(self, t):
        s, d = t.shape
        d_pad = -(-d // INT4_GROUP) * INT4_GROUP
        if d_pad != d:
            t = torch.nn.functional.pad(t, (0, d_pad - d))
        g = t.view(s, d_pad // INT4_GROUP, INT4_GROUP).float()
        scales = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
        q = torch.round(g / scales).clamp(-7, 7).to(torch.int8) + 7  # [0, 14]
        q = q.view(s, d_pad).to(torch.uint8)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
        return packed, scales.view(s, d_pad // INT4_GROUP).to(torch.float16).contiguous()

    def dequantize(self, payload, scales, out):
        s, half = payload.shape
        d_pad = half * 2
        q = torch.empty((s, d_pad), dtype=torch.uint8, device=payload.device)
        q[:, 0::2] = payload & 0xF
        q[:, 1::2] = payload >> 4
        v = (q.to(torch.float32) - 7.0).view(s, d_pad // INT4_GROUP, INT4_GROUP)
        v = v * scales.to(torch.float32).unsqueeze(-1)
        out.copy_(v.view(s, d_pad)[:, : out.shape[1]])
        return out

    def payload_shape(self, s, d):
        d_pad = -(-d // INT4_GROUP) * INT4_GROUP
        return (s, d_pad // 2), torch.uint8

    def scales_shape(self, s, d):
        d_pad = -(-d // INT4_GROUP) * INT4_GROUP
        return (s, d_pad // INT4_GROUP), torch.float16


CODECS = {"int4": _CodecINT4(), "fp8": _CodecFP8(), "bf16": _CodecBF16()}


def _quantize_to_host(codec, t):
    """Quantize `t` in row chunks on its own device, landing the result in pageable RAM.

    Pageable, not pinned: AudioRefine measured the pinned host allocator
    retaining freed blocks across rebuilds (their PR #1).
    """
    s, d = t.shape
    p_shape, p_dtype = codec.payload_shape(s, d)
    s_shape, s_dtype = codec.scales_shape(s, d)
    payload = torch.empty(p_shape, dtype=p_dtype, device="cpu")
    scales = None if s_shape is None else torch.empty(s_shape, dtype=s_dtype, device="cpu")
    for a in range(0, s, QUANT_CHUNK_ROWS):
        b = min(a + QUANT_CHUNK_ROWS, s)
        pq, ps = codec.quantize(t[a:b])
        payload[a:b].copy_(pq)
        if scales is not None:
            scales[a:b].copy_(ps)
    return payload, scales


def _dequantize_from_host(codec, payload, scales, out):
    for a in range(0, payload.shape[0], QUANT_CHUNK_ROWS):
        b = min(a + QUANT_CHUNK_ROWS, payload.shape[0])
        p = payload[a:b].to(out.device)
        sc = None if scales is None else scales[a:b].to(out.device)
        codec.dequantize(p, sc, out[a:b])
    return out


# ---------------------------------------------------------------------------
# live-row bookkeeping

def _dilate(grid, tokens):
    """A [t, h, w] bool grid widened by `tokens` cells along every axis."""
    if tokens <= 0:
        return grid
    k = 2 * int(tokens) + 1
    pooled = torch.nn.functional.max_pool3d(grid[None, None].to(torch.float32), k, stride=1, padding=int(tokens))
    return pooled[0, 0] > 0.5


class _Rows:
    """Which packed rows one call computes, and which of them it regenerates.

    `live` is every row a cached step computes. `regen` is the video token
    grid of rows the sampler regenerates and `audio` the audio rows it does:
    `live` holds those, the always-live kinds and the halo. `text` marks the
    text rows, for `verify`'s text line. `sig` names the live set, so a slot
    built for other rows is not reused.
    """

    def __init__(self, live, regen, audio, text, halo_rows):
        self.live, self.regen, self.audio, self.text = live, regen, audio, text
        self.halo_rows = int(halo_rows)
        self.n_live = int(live.sum())
        self.share = self.n_live / max(int(live.numel()), 1)
        self.frozen_audio = not bool(audio.all())
        self.sig = hashlib.blake2b(live.numpy().tobytes(), digest_size=8).hexdigest()


def _rows(layout, video_mask, audio_mask, halo=0):
    """The live rows of a call from the masks core hands the model, or None when the layout does not fit them.

    A video row is read with core's own pooling, so it is frozen exactly when
    `_forward` pins it. No mask for a stream means the stream is fully open
    (`MiniMaxH3._denoise_mask_values` passes one only when a row is below 1).
    """
    _text_len, latent_t, lat_h, lat_w, _audio_t = layout.signature
    grid = (latent_t, lat_h // 2, lat_w // 2)
    regen = torch.ones(grid, dtype=torch.bool)
    if video_mask is not None:
        # Core's pooling raises on a mask of another length and pads or crops one of
        # another height or width, so a mask that is not on the layout's grid is
        # turned away here. That includes an odd-sized latent, whose mask is one
        # short of the padded layout: core pads its video circularly and its mask by
        # replication, so a frozen edge token there can carry content that changes.
        if tuple(video_mask.shape[2:]) != (latent_t, lat_h, lat_w):
            return None
        m = mm_h3.mask_row_values(video_mask[0, 0].to(torch.float32), latent_t, lat_h, lat_w)
        if m is not None:
            if m.numel() != regen.numel():
                return None
            regen = (m >= FROZEN_BELOW).reshape(grid).cpu()
    live = torch.zeros(layout.seq_len, dtype=torch.bool)
    text = torch.zeros(layout.seq_len, dtype=torch.bool)
    audio = None
    halo_rows = 0
    for a, b, kind in layout.segments:
        if kind == "text":
            text[a:b] = True
        if kind in ALWAYS_LIVE_KINDS:
            live[a:b] = True
        elif kind == "video":
            if b - a != regen.numel():
                return None
            widened = _dilate(regen, halo)
            halo_rows = int(widened.sum()) - int(regen.sum())
            live[a:b] = widened.reshape(-1)
        elif kind == "audio":
            audio = torch.ones(b - a, dtype=torch.bool)
            if audio_mask is not None:
                am = audio_mask[0, 0].to(torch.float32).reshape(-1)
                if am.numel() != b - a:
                    return None
                audio = (am >= FROZEN_BELOW).cpu()
            live[a:b] = audio
    if audio is None:
        return None
    return _Rows(live, regen, audio, text, halo_rows)


def _live_segments(mod_segments, idx):
    """`mod_segments` restricted to the rows in `idx` and renumbered onto them.

    `idx` is the sorted packed-row index of the live rows. A segment's row is
    an int or a per-row LongTensor (core passes one for a stream whose rows
    run at different timesteps); a tensor row is gathered at the live rows it
    holds. Returns None when the segments do not cover the live rows exactly,
    which means core's layout moved under this module.
    """
    out = []
    covered = 0
    for a, b, row in mod_segments:
        lo = int(torch.searchsorted(idx, a))
        hi = int(torch.searchsorted(idx, b))
        if hi > lo:
            r = row[idx[lo:hi] - a] if isinstance(row, torch.Tensor) else row
            out.append((lo, hi, r))
            covered += hi - lo
    if covered != idx.numel():
        return None
    return out


def _pins(layout, t_v, t_a, payload, frozen_audio=False):
    """The timesteps the cached rows run at, per `MiniMaxH3Model._forward`'s `seg_t`.

    Recorded at the build and compared on every cached call: if any moved, the
    cache holds rows modulated at a timestep they no longer run at.
    """
    kinds = {k for _, _, k in layout.segments}
    vis_aug = float(payload.get("visual_cond_noise_aug", mm_h3.VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", mm_h3.AUDIO_COND_TIMESTEP))
    pins = {"video": max(t_v, mm_h3.VISUAL_COND_TIMESTEP)}
    if frozen_audio:
        # Core's `t_pin_a`. Constant while `AUDIO_COND_TIMESTEP` is 1.0, so nothing
        # can make it move today and no check can; it is here for the day core's
        # constant is lower.
        pins["audio"] = max(t_a, mm_h3.AUDIO_COND_TIMESTEP)
    if kinds & {"cond", "ref_img"}:
        pins["cond"] = max(t_v, vis_aug)
    if kinds & {"cond_audio", "ref_audio"}:
        pins["cond_audio"] = max(t_a, aud_aug)
    return pins


# ---------------------------------------------------------------------------
# state

class _Slot:
    """One kept forward: the per-block hidden states of one conditioning's build."""

    def __init__(self, codec, n_blocks, layout_sig, pins, rows_sig, content):
        self.codec = codec
        self.h = [None] * n_blocks          # (payload, scales) in RAM
        self.layout_sig = layout_sig
        self.pins = pins
        self.rows_sig = rows_sig
        self.content = content              # `_content_sample` at the build
        self.complete = False
        self.steps_since_build = 0
        self.dq = None                      # device dequant buffer, per call

    def nbytes(self):
        n = 0
        for entry in self.h:
            if entry is not None:
                n += sum(t.numel() * t.element_size() for t in entry if t is not None)
        return n


class _State:
    def __init__(self, dm, precision, refresh, refresh_every, verify, halo=0):
        self.dm = dm
        self.n_blocks = len(dm.blocks)
        self.codec = CODECS[precision]
        self.refresh = refresh
        self.refresh_every = refresh_every
        self.verify = verify
        self.halo = int(halo)
        self.slots = {}
        self.replaces = {}
        self.mode = "off"                   # off | build | cached, per call
        self.slot = None
        self.slot_key = None
        self.rows = None                    # this call's `_Rows`
        self.live = None                    # the rows this pass computes: `rows.live`, or less for verify's text line
        self.live_idx = None
        self.live_segs = None
        self.stages = None                  # seconds by stage of a cached pass, while verify times one
        self.counts = {"build": 0, "cached": 0, "stock": 0}
        self.calls = deque(maxlen=64)       # per-call records, read by the check
        self.verify_log = deque(maxlen=64)
        self.warned = set()

    def warn_once(self, tag, msg):
        if tag not in self.warned:
            self.warned.add(tag)
            log.warning("[h3] frozen video cache: %s", msg)

    def free(self):
        self.slots.clear()
        self.slot = None


def _foreign_patch(state, transformer_options):
    """Why this call cannot take the cached path, or None."""
    dit = transformer_options.get("patches_replace", {}).get("dit", {})
    for key, fn in dit.items():
        if state.replaces.get(key) is not fn:
            return (f"patches_replace['dit'][{key!r}] is not this node's; a cached "
                    f"step would skip it")
    for i, blk in enumerate(state.dm.blocks):
        if "forward" in vars(blk) or "forward" in vars(blk.attn):
            return (f"block {i}'s forward or attention forward is object-patched "
                    f"(sage capture, exact_blocks); a cached step would skip it")
    return None


# ---------------------------------------------------------------------------
# block paths

def _qkv_rope(attn, h, rope_freqs):
    """Core's `Attention.forward` up to the attention call: qkv, per-head RMSNorm, rope."""
    s = h.shape[0]
    q, k, v = attn.qkv_proj(h).split(attn.heads * attn.head_dim, dim=-1)
    v = v.view(s, attn.heads, attn.head_dim)
    q = q.view(1, s, attn.heads, attn.head_dim)
    k = k.view(1, s, attn.heads, attn.head_dim)
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=h.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=h.device)
    rot = rope_freqs.shape[-3] * 2
    comfy.quant_ops.ck.rms_rope_split_half_(
        q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
    return q[0], k[0], v


#: The stages `verify` splits a cached step into, in the order a block meets
#: them. `between blocks` is everything outside this module's block: core's
#: loop and prefetch, the embedding before the first block and the final
#: layer after the last.
STAGES = ("between blocks", "live rows", "store to card", "qkv, every row", "attention")


def _lap(state, name, device):
    """Charge the time since the last lap to `name`. A no-op unless `verify` is timing a cached pass.

    It waits for the card each time, which is why an unverified run never
    does it: the stage split comes from the verify arm and the step's cost
    from the plain one.
    """
    st = state.stages
    if st is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    now = time.perf_counter()
    st[name] = st.get(name, 0.0) + now - st["_t"]
    st["_t"] = now


def _build_block(state, i, args, extra):
    blk = state.dm.blocks[i]
    slot = state.slot

    def capture(h, rope_freqs=None, transformer_options={}):
        slot.h[i] = _quantize_to_host(slot.codec, h)
        return blk.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)

    return extra["original_block"](dict(args, attention=capture))


def _cached_block(state, i, args):
    """Core's `DiTBlock.forward` on the live rows, attending over the kept rows."""
    blk = state.dm.blocks[i]
    attn = blk.attn
    x = args["img"]
    rope_freqs = args["rope_freqs"]
    idx, segs, slot = state.live_idx, state.live_segs, state.slot
    _lap(state, "between blocks", x.device)

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = blk.adaln_proj(args["t_emb"])
    xl = x.index_select(0, idx)
    h = mm_h3._mod_scale_shift(blk.norm1(xl), shift_msa, scale_msa, segs)
    _lap(state, "live rows", x.device)

    if slot.dq is None or slot.dq.dtype != x.dtype or slot.dq.device != x.device:
        slot.dq = torch.empty((x.shape[0], h.shape[1]), dtype=x.dtype, device=x.device)
    payload, scales = slot.h[i]
    hf = _dequantize_from_host(slot.codec, payload, scales, slot.dq)
    hf.index_copy_(0, idx, h.to(hf.dtype))
    _lap(state, "store to card", x.device)
    q, k, v = _qkv_rope(attn, hf, rope_freqs)
    q = q.index_select(0, idx)
    _lap(state, "qkv, every row", x.device)

    q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    # The options go in as the block got them, attention override included. A
    # cached step's queries are the live rows against every row's K/V; Sol
    # declines that (a q/k length mismatch) and hands it to the override it
    # was installed on, which is where the graph's dense backend lives (core's
    # Model Attention Backend node: the kitchen kernel). Until 2026-10-07 the
    # override was removed here, the call fell to core's default, torch's
    # SDPA, and a cached step cost more than a stock one
    # (`bench/results/2026-10-07_frozen_cache_rectangle_kernel.md`).
    out = optimized_attention(q, k, v, attn.heads, preferred_attention=attn.comfy_attention,
                              mask=None, skip_reshape=True,
                              transformer_options=args["transformer_options"])
    _lap(state, "attention", x.device)
    xl = mm_h3._mod_gate(xl, gate_msa, attn.out_proj(out.squeeze(0)), segs)
    h = mm_h3._mod_scale_shift(blk.norm2(xl), shift_mlp, scale_mlp, segs)
    xl = mm_h3._mod_gate(xl, gate_mlp, blk.mlp(h), segs)
    x.index_copy_(0, idx, xl)
    _lap(state, "live rows", x.device)
    return {"img": x}


def _make_block_replace(state, i):
    def replace(args, extra):
        if i == 0 and state.mode != "off":
            _first_block(state, args)
        if state.mode == "build":
            state.counts["build"] += 1
            return _build_block(state, i, args, extra)
        if state.mode == "cached":
            state.counts["cached"] += 1
            return _cached_block(state, i, args)
        state.counts["stock"] += 1
        return extra["original_block"](args)
    return replace


def _first_block(state, args):
    """Per-call setup that needs the block arguments: live rows and their segments."""
    rows = args["img"].shape[0]
    device = args["img"].device
    idx = segs = None
    if state.live.numel() == rows:
        idx = state.live.nonzero().flatten().to(device)
        segs = _live_segments(args["mod_segments"], idx)
    if segs is None or (state.mode == "cached" and state.slot.h[0][0].shape[0] != rows):
        state.warn_once("segments", "the packed layout does not match what this "
                        "module expects; running the stock path")
        if state.mode == "build":
            state.slots.pop(state.slot_key, None)
        state.mode = "off"
        return
    state.live_idx = idx
    state.live_segs = segs


# ---------------------------------------------------------------------------
# wrappers

def _timesteps(state, timestep, transformer_options):
    dm = state.dm
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", dm.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", dm.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - mm_h3.time_shift_sigma(sigma_v, shift_v, shift_a))
    return t_v, t_a


def _gate(x, kwargs, transformer_options, halo=0):
    """(layout, rows, None) for a call the cache takes, else (None, rows or None, why not).

    It takes a call on a packed layout that freezes some video or audio row,
    regenerates some other, and whose live rows are at most
    `LIVE_SHARE_LIMIT` of the sequence. The refine pass (video frozen whole,
    audio open) and a masked window (audio and most video frozen) are both
    that.
    """
    layout = (kwargs.get("minimax_payload") or {}).get("layout")
    video_mask = kwargs.get("denoise_mask")
    audio_mask = kwargs.get("audio_denoise_mask")
    if layout is None or not isinstance(x, (list, tuple)) or len(x) != 2:
        return None, None, "no packed layout"
    # A context window runs the model on one slice of the latent after another
    # inside a single sampling run. A slot is found by conditioning index and
    # layout shape, which two slices share, so it is refused rather than served
    # one slice's rows for another's.
    if transformer_options.get("context_window") is not None:
        return None, None, "a context window"
    # Core passes a mask only when some row of its stream is below 1
    # (`MiniMaxH3._denoise_mask_values`), so no mask means fully open.
    if video_mask is None and audio_mask is None:
        return None, None, "no mask"
    kinds = {k for _, _, k in layout.segments}
    if not {"video", "audio"} <= kinds:
        return None, None, "no video or audio rows"
    rows = _rows(layout, video_mask, audio_mask, halo)
    if rows is None:
        return None, None, "the masks do not fit the layout"
    if bool(rows.regen.all()) and not rows.frozen_audio:
        return None, rows, "nothing frozen"
    if not bool(rows.regen.any()) and not bool(rows.audio.any()):
        return None, rows, "nothing regenerates"
    if rows.share > LIVE_SHARE_LIMIT:
        return None, rows, "live share above the limit"
    return layout, rows, None


def _content_sample(x, rows, payload):
    """A strided sample of what the kept rows are made from, to be compared exactly with the build's.

    The premise of the cache is that a kept row's input is the same on every
    step. A slot is found by conditioning index, layout shape and live rows,
    none of which names the content: another window, or another reference
    under the same shape, would be served the first one's rows. So the video
    latent under every token the sampler keeps, and every conditioning latent
    in the payload, are sampled at the build and on each cached call.
    Frozen audio is left out: core rescales it by the step's sigma on the way
    in and back, which need not round the same twice.
    """
    video = x[0]
    parts = []
    kept = ~_regen_latent(rows, video)
    if bool(kept.any()):
        cells = video[0][:, kept.to(video.device)]
        parts.append(cells[:, ::max(1, cells.shape[1] // CONTENT_SAMPLE)].flatten())
    for name in ("cond_video_latents", "cond_audio_latents"):
        for z in payload.get(name) or ():
            flat = z.flatten()
            parts.append(flat[::max(1, flat.numel() // CONTENT_SAMPLE)].to(device=video.device, dtype=video.dtype))
    return torch.cat(parts) if parts else None


def _same_content(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and bool(torch.equal(a, b))


def _regen_latent(rows, video):
    """`rows.regen` as a [t, h, w] bool mask over `video`'s own latent grid."""
    g = rows.regen.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    return g[:video.shape[2], :video.shape[3], :video.shape[4]]


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


def _compare(out, ref, rows):
    """A cached step against the same step run stock, on the rows the sampler keeps from it.

    Video: cosine and relative L2 over the regenerated rows, and the L2 again
    on the ring (`RING_TOKENS` from a kept row) and on the interior. Audio:
    the same pair over its live rows. Kept rows are left out: core multiplies
    their output by a mask of 0.
    """
    got = {}
    if bool(rows.regen.any()):
        a, b = out[0][0].float(), ref[0][0].float()
        kept_near = _dilate(~rows.regen, RING_TOKENS)
        ring = rows.regen & kept_near
        inner = rows.regen & ~kept_near
        whole = _regen_latent(rows, out[0]).to(a.device)
        va, vb = a[:, whole].flatten(), b[:, whole].flatten()
        got["video_cos"] = float(torch.nn.functional.cosine_similarity(va, vb, dim=0))
        got["video_rel_l2"] = _rel(va, vb)
        for name, grid in (("ring", ring), ("interior", inner)):
            got[f"{name}_rows"] = int(grid.sum())
            if got[f"{name}_rows"]:
                sel = _regen_latent(_Rows(rows.live, grid, rows.audio, rows.text, 0), out[0]).to(a.device)
                got[f"{name}_rel_l2"] = _rel(a[:, sel].flatten(), b[:, sel].flatten())
    if bool(rows.audio.any()):
        a, b = out[1][0].float(), ref[1][0].float()
        sel = rows.audio.reshape(a.shape[1:]).to(a.device)
        aa, ab = a[:, sel].flatten(), b[:, sel].flatten()
        got["audio_cos"] = float(torch.nn.functional.cosine_similarity(aa, ab, dim=0))
        got["audio_rel_l2"] = _rel(aa, ab)
    return got


def _make_diffusion_wrapper(state):
    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        state.mode = "off"
        state.counts = {"build": 0, "cached": 0, "stock": 0}
        record = {"mode": "off", "reason": None}
        try:
            layout, rows, why = _gate(x, kwargs, transformer_options, state.halo)
            record["reason"] = why
            if rows is not None:
                record.update(rows=int(rows.live.numel()), live_rows=rows.n_live, live_sig=rows.sig,
                              live_share=rows.share, halo_rows=rows.halo_rows)
            if layout is not None:
                foreign = _foreign_patch(state, transformer_options)
                if foreign is not None:
                    state.warn_once("foreign", foreign + "; running the stock path")
                    layout = None
                    record["reason"] = "foreign patch"
            if layout is None:
                # A slot built earlier in this run stays in RAM through a declined call
                # and is used again, rightly, if a later call matches it; the run's end
                # frees it either way.
                if why == "live share above the limit":
                    state.warn_once(("share", rows.sig),
                                    f"{rows.n_live} of {rows.live.numel()} rows are live, more than "
                                    f"{LIVE_SHARE_LIMIT:g} of the sequence; running the stock path")
                out = executor(x, timestep, context, transformer_options, **kwargs)
                record["counts"] = dict(state.counts)
                state.calls.append(record)
                return out

            payload = kwargs.get("minimax_payload") or {}
            t_v, t_a = _timesteps(state, timestep, transformer_options)
            pins = _pins(layout, t_v, t_a, payload, rows.frozen_audio)
            content = _content_sample(x, rows, payload)
            key = (tuple(transformer_options.get("cond_or_uncond") or ()), layout.signature)
            slot = state.slots.get(key)
            reason = None
            if slot is None or not slot.complete:
                reason = "no cache yet"
            elif slot.layout_sig != layout.signature:
                reason = "layout changed"
            elif slot.rows_sig != rows.sig:
                reason = "the frozen rows changed"
            elif slot.pins != pins:
                reason = "a cached row's timestep moved"
            elif not _same_content(slot.content, content):
                reason = "the kept rows' input changed"
            elif state.refresh and slot.steps_since_build >= state.refresh_every:
                reason = "refresh"
            if reason is not None:
                slot = _Slot(state.codec, state.n_blocks, layout.signature, pins, rows.sig, content)
                state.slots[key] = slot
                state.mode = "build"
            else:
                slot.steps_since_build += 1
                state.mode = "cached"
            state.slot, state.slot_key = slot, key
            state.rows, state.live = rows, rows.live
            record["reason"] = reason
            record["sigma"] = 1.0 - t_v

            ref = None
            if state.mode == "cached" and state.verify:
                mode, state.mode = state.mode, "off"
                ref = executor(x, timestep, context, transformer_options, **kwargs)
                ref = [t.detach().clone() for t in ref]
                state.mode = mode
                state.counts = {"build": 0, "cached": 0, "stock": 0}
                if x[0].device.type == "cuda":
                    torch.cuda.synchronize(x[0].device)
                state.stages = {"_t": time.perf_counter()}

            t0 = time.perf_counter()
            try:
                out = executor(x, timestep, context, transformer_options, **kwargs)
            except BaseException:
                if state.mode == "build":
                    state.slots.pop(key, None)
                raise
            finally:
                slot.dq = None
            _lap(state, "between blocks", x[0].device)
            if x[0].device.type == "cuda":
                torch.cuda.synchronize(x[0].device)
            record["mode"] = state.mode
            record["counts"] = dict(state.counts)
            record["seconds"] = time.perf_counter() - t0
            if state.stages is not None:
                if state.mode == "cached":
                    record["stages"] = {k: state.stages.get(k, 0.0) for k in STAGES}
                state.stages = None
            if state.mode == "build":
                slot.complete = all(e is not None for e in slot.h)
                record["cache_bytes"] = slot.nbytes()
                log.info("[h3] frozen video cache: built (%s), %d of %d rows live (%d of them halo), "
                         "%s, %.2f GiB in RAM", reason, rows.n_live, layout.seq_len, rows.halo_rows,
                         slot.codec.name, slot.nbytes() / 2**30)
            if ref is not None and state.mode == "cached":
                verify = _compare(out, ref, rows)
                if bool(rows.text.any()):
                    # The same step once more with the text rows cached too: how much
                    # of the departure text being live buys. The output is not used.
                    kept = [t.detach().clone() for t in out]
                    state.live = rows.live & ~rows.text
                    counts = dict(state.counts)
                    try:
                        dead = executor(x, timestep, context, transformer_options, **kwargs)
                    finally:
                        slot.dq = None
                        state.live = rows.live
                        state.counts = counts
                    if state.mode == "cached":
                        text = _compare(dead, ref, rows)
                        for k in ("video_rel_l2", "audio_rel_l2"):
                            if k in text:
                                verify[f"text_cached_{k}"] = text[k]
                    state.mode = "cached"
                    out = kept
                record["verify"] = verify
                state.verify_log.append(verify)
                log.info("[h3] frozen video cache: verify, sigma %.4f, %d of %d rows live, against the "
                         "stock step: %s", 1.0 - t_v, rows.n_live, layout.seq_len,
                         ", ".join(f"{k} {v:.6g}" for k, v in verify.items()))
                if "stages" in record:
                    log.info("[h3] frozen video cache: cached step %.2f s: %s", record["seconds"],
                             ", ".join(f"{k} {v:.2f} s" for k, v in record["stages"].items()))
            state.calls.append(record)
            return out
        finally:
            state.mode = "off"
            state.slot = None
            state.rows = state.live = None
            state.stages = None
    return wrapper


def _make_outer_sample_wrapper(state):
    """Scope the cache to one sampling run, and keep the malloc graph out of it.

    The compiler flip is **inherited** from AudioRefine and not reproduced
    here: their account is that core records the diffusion forward into an
    aimdo malloc graph that assumes a repeatable allocation pattern, and the
    build and cached steps allocate differently. It mutates a process-global,
    so it is set only while this model samples, only when the graph is on,
    and restored in `finally`.
    """
    def wrapper(executor, *args, **kwargs):
        import comfy.cli_args
        cli = comfy.cli_args.args
        flipped = False
        try:
            device = comfy.model_management.get_torch_device()
            if comfy.model_prefetch.malloc_graph_enabled(device):
                cli.disable_comfy_compiler = True
                flipped = True
                state.warn_once("compiler", "the comfy compiler's malloc graph is off "
                                "while this model samples, and restored after")
        except Exception as e:  # a core without the graph: nothing to flip
            state.warn_once("compiler_probe", f"could not probe the malloc graph ({e})")
        try:
            return executor(*args, **kwargs)
        finally:
            if flipped:
                cli.disable_comfy_compiler = False
            state.free()
    return wrapper


def attach(model, precision="int4", refresh=False, refresh_every=2, verify=False, halo=0):
    """Clone `model` with the cache attached. The node's body, and the check's entry."""
    dm = model.get_model_object("diffusion_model")
    if not isinstance(dm, mm_h3.MiniMaxH3Model):
        raise ValueError("MiniMaxH3FrozenVideoCache needs a MiniMax H3 model")
    if precision == "fp8" and _FP8 is None:
        raise ValueError("this PyTorch has no float8_e4m3fn; use int4 or bf16")
    existing = model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
    if existing:
        raise ValueError(
            f"this model already carries patches_replace['dit'] entries ({sorted(existing)[:3]}...): "
            f"core's sparse attention or FunControl. A cached step would skip "
            f"them, so the cache refuses to compose. Put it on a model without them.")
    m = model.clone()
    if int(halo) < 0:
        raise ValueError(f"halo is a number of tokens and cannot be {halo}")
    state = _State(dm, precision, refresh, refresh_every, verify, halo)
    for i in range(state.n_blocks):
        fn = _make_block_replace(state, i)
        state.replaces[("double_block", i)] = fn
        m.set_model_patch_replace(fn, "dit", "double_block", i)
    m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, _make_diffusion_wrapper(state))
    m.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, _make_outer_sample_wrapper(state))
    m._h3_frozen_video_cache = state
    return m


class MiniMaxH3FrozenVideoCache(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FrozenVideoCache",
            display_name="MiniMax H3 Frozen Video Cache",
            category="MiniMax H3/audio",
            description=(
                "For a pass that keeps most of its rows: an audio-only refine pass "
                "(MiniMax H3 Audio Refine Mask) or a masked video-to-video window. The first "
                "step runs stock and keeps each block's attention input in RAM; later steps "
                "compute only the rows being generated and the text against it. A call that "
                "freezes nothing, or whose live rows are too much of the sequence to gain, "
                "runs stock and the log says why. The kept rows stop reacting to the "
                "generated ones, which `verify` measures. Ported from ComfyUI-H3-AudioRefine "
                "(MIT)."),
            inputs=[
                io.Model.Input("model", tooltip="The pass's model. Sol-Attn and a LoRA applied at the "
                                                "call can already be on it."),
                io.Combo.Input("precision", options=list(CODECS), default="int4",
                               tooltip="How the kept hidden states are stored in RAM. int4 is the "
                                       "smallest; bf16 is exact to the model's compute dtype."),
                io.Boolean.Input("refresh", default=False, advanced=True,
                                 tooltip="Rebuild the cache every `refresh_every` cached steps."),
                io.Int.Input("refresh_every", default=2, min=1, max=100, advanced=True,
                             tooltip="Cached steps between rebuilds, when `refresh` is on."),
                io.Boolean.Input("verify", default=False, advanced=True,
                                 tooltip="Also run each cached step stock and log how far the "
                                         "generated rows are from it, the mask's edge apart from "
                                         "its inside, and where the cached step's time went. "
                                         "Costs more than a full step each."),
                # 0 is a width, not a mode: no kept row is recomputed, and nothing else
                # in this module switches on it.
                io.Int.Input("halo", default=0, min=0, max=16, optional=True, advanced=True,
                             tooltip="Kept video tokens this close to a regenerated one are "
                                     "recomputed on every step, so the subject is drawn against "
                                     "neighbours that have seen it. They are never regenerated. "
                                     "Each token of width costs time, and the added rows count "
                                     "toward the share at which a window is left uncached, so a "
                                     "wide halo can switch the cache off for it."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, precision="int4", refresh=False, refresh_every=2,
                verify=False, halo=0) -> io.NodeOutput:
        return io.NodeOutput(attach(model, precision, refresh, refresh_every, verify, halo))
