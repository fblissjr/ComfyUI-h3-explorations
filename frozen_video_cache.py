"""Cache the frozen video of an H3 audio-refine pass, so its steps run on the live rows only.

The refine pass (`audio_refine.py`, `h3_config.AUDIO_REFINE`) keeps the video
exactly and reopens the audio. Every step still runs the whole packed sequence,
and the video is nearly all of it. Under a video mask of 0 the video rows see
the same input on every step (`comfy/model_base.py::MiniMaxH3.scale_latent_inpaint`
injects the same clean-plus-noise blend each call) at the same pinned timestep
(`comfy/ldm/minimax/model.py::MiniMaxH3Model._forward`, `t_pin_v`). What still
moves between steps is the audio and the text: text rows follow the video
stream's timestep, which falls every step.

So the first model call of the pass (the build) runs every block stock and
keeps each block's attention input, the post-norm modulated hidden state. Each
later call (a cached step) computes only the live rows, text and audio, and
their queries attend against K/V rebuilt from the kept hidden state with the
live rows written in. The cached rows stop reacting to the new audio and text:
that is the approximation, and the `verify` switch measures it on a render.

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
- **One sampling run, never longer.** The store is freed when the pass ends.
  AudioRefine keeps it across prompts behind a sum/abs-sum fingerprint, which
  nothing shows can tell two seeds of one prompt apart.
- **Refuses what it would silently skip**: a foreign `patches_replace["dit"]`
  entry (core VSA, taomate, FunControl) at patch time, and an object-patched
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

#: Rows recomputed on every cached step. Everything else is cached.
#: **Reasoned** from `MiniMaxH3Model._forward`'s `seg_t`: text follows `t_v`,
#: audio is what the pass generates; video and cond rows pin at the cond
#: timestep, and cond/ref audio at `AUDIO_COND_TIMESTEP`.
LIVE_KINDS = ("text", "audio")

#: Gate thresholds on the pooled masks core hands the model. **Inherited**
#: from AudioRefine's gate.
FROZEN_BELOW = 1e-3

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

def _live_spans(layout):
    return [(a, b) for a, b, kind in layout.segments if kind in LIVE_KINDS]


def _live_segments(mod_segments, spans):
    """`mod_segments` restricted to the live rows and renumbered onto them.

    A segment's row is an int or a per-row LongTensor; a tensor row is sliced
    with its span. Returns None when the segments do not cover the live rows
    exactly, which means core's layout moved under this module.
    """
    out = []
    off = 0
    for la, lb in spans:
        covered = 0
        for a, b, row in mod_segments:
            s, e = max(a, la), min(b, lb)
            if s < e:
                r = row[s - a:e - a] if isinstance(row, torch.Tensor) else row
                out.append((off + s - la, off + e - la, r))
                covered += e - s
        if covered != lb - la:
            return None
        off += lb - la
    return out


def _pins(layout, t_v, t_a, payload):
    """The timesteps the cached rows run at, per `MiniMaxH3Model._forward`'s `seg_t`.

    Recorded at the build and compared on every cached call: if any moved, the
    cache holds rows modulated at a timestep they no longer run at.
    """
    kinds = {k for _, _, k in layout.segments}
    vis_aug = float(payload.get("visual_cond_noise_aug", mm_h3.VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", mm_h3.AUDIO_COND_TIMESTEP))
    pins = {"video": max(t_v, mm_h3.VISUAL_COND_TIMESTEP)}
    if kinds & {"cond", "ref_img"}:
        pins["cond"] = max(t_v, vis_aug)
    if kinds & {"cond_audio", "ref_audio"}:
        pins["cond_audio"] = max(t_a, aud_aug)
    return pins


# ---------------------------------------------------------------------------
# state

class _Slot:
    """One kept forward: the per-block hidden states of one conditioning's build."""

    def __init__(self, codec, n_blocks, layout_sig, pins):
        self.codec = codec
        self.h = [None] * n_blocks          # (payload, scales) in RAM
        self.layout_sig = layout_sig
        self.pins = pins
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
    def __init__(self, dm, precision, refresh, refresh_every, verify):
        self.dm = dm
        self.n_blocks = len(dm.blocks)
        self.codec = CODECS[precision]
        self.refresh = refresh
        self.refresh_every = refresh_every
        self.verify = verify
        self.slots = {}
        self.replaces = {}
        self.mode = "off"                   # off | build | cached, per call
        self.slot = None
        self.slot_key = None
        self.live_idx = None
        self.live_segs = None
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


def _dense_options(transformer_options):
    """The options a cached step's attention runs with: no override.

    A cached step's queries are the live rows against every row's K/V. Sol
    declines a q/k length mismatch to dense anyway, and a sparse pattern over
    a few thousand queries buys nothing, so the call goes straight to the
    model's own backend (`preferred_attention`).
    """
    opts = dict(transformer_options)
    opts.pop("optimized_attention_override", None)
    return opts


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

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = blk.adaln_proj(args["t_emb"])
    xl = x.index_select(0, idx)
    h = mm_h3._mod_scale_shift(blk.norm1(xl), shift_msa, scale_msa, segs)

    if slot.dq is None or slot.dq.dtype != x.dtype or slot.dq.device != x.device:
        slot.dq = torch.empty((x.shape[0], h.shape[1]), dtype=x.dtype, device=x.device)
    payload, scales = slot.h[i]
    hf = _dequantize_from_host(slot.codec, payload, scales, slot.dq)
    hf.index_copy_(0, idx, h.to(hf.dtype))
    q, k, v = _qkv_rope(attn, hf, rope_freqs)
    q = q.index_select(0, idx)

    q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    out = optimized_attention(q, k, v, attn.heads, preferred_attention=attn.comfy_attention,
                              mask=None, skip_reshape=True,
                              transformer_options=_dense_options(args["transformer_options"]))
    xl = mm_h3._mod_gate(xl, gate_msa, attn.out_proj(out.squeeze(0)), segs)
    h = mm_h3._mod_scale_shift(blk.norm2(xl), shift_mlp, scale_mlp, segs)
    xl = mm_h3._mod_gate(xl, gate_mlp, blk.mlp(h), segs)
    x.index_copy_(0, idx, xl)
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
    layout = args["layout"]
    spans = _live_spans(layout)
    segs = _live_segments(args["mod_segments"], spans)
    rows = args["img"].shape[0]
    if segs is None or (state.mode == "cached" and state.slot.h[0][0].shape[0] != rows):
        state.warn_once("segments", "the packed layout does not match what this "
                        "module expects; running the stock path")
        if state.mode == "build":
            state.slots.pop(state.slot_key, None)
        state.mode = "off"
        return
    device = args["img"].device
    state.live_idx = torch.cat([torch.arange(a, b, device=device) for a, b in spans])
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


def _gate(x, kwargs):
    """True when this call is a frozen-video, generating-audio call on a packed layout."""
    layout = (kwargs.get("minimax_payload") or {}).get("layout")
    video_mask = kwargs.get("denoise_mask")
    audio_mask = kwargs.get("audio_denoise_mask")
    if layout is None or video_mask is None or not isinstance(x, (list, tuple)) or len(x) != 2:
        return None
    if float(video_mask.max()) >= FROZEN_BELOW:
        return None
    # Core passes audio_denoise_mask only when some audio row is below 1
    # (`MiniMaxH3._denoise_mask_values`), so None means the audio is fully open.
    if audio_mask is not None and float(audio_mask.max()) <= FROZEN_BELOW:
        return None
    kinds = {k for _, _, k in layout.segments}
    if not {"video", "audio"} <= kinds:
        return None
    return layout


def _make_diffusion_wrapper(state):
    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        state.mode = "off"
        state.counts = {"build": 0, "cached": 0, "stock": 0}
        record = {"mode": "off", "reason": None}
        try:
            layout = _gate(x, kwargs)
            if layout is not None:
                foreign = _foreign_patch(state, transformer_options)
                if foreign is not None:
                    state.warn_once("foreign", foreign + "; running the stock path")
                    layout = None
                    record["reason"] = "foreign patch"
            if layout is None:
                out = executor(x, timestep, context, transformer_options, **kwargs)
                record["counts"] = dict(state.counts)
                state.calls.append(record)
                return out

            payload = kwargs.get("minimax_payload") or {}
            t_v, t_a = _timesteps(state, timestep, transformer_options)
            pins = _pins(layout, t_v, t_a, payload)
            key = (tuple(transformer_options.get("cond_or_uncond") or ()), layout.signature)
            slot = state.slots.get(key)
            reason = None
            if slot is None or not slot.complete:
                reason = "no cache yet"
            elif slot.layout_sig != layout.signature:
                reason = "layout changed"
            elif slot.pins != pins:
                reason = "a cached row's timestep moved"
            elif state.refresh and slot.steps_since_build >= state.refresh_every:
                reason = "refresh"
            if reason is not None:
                slot = _Slot(state.codec, state.n_blocks, layout.signature, pins)
                state.slots[key] = slot
                state.mode = "build"
            else:
                slot.steps_since_build += 1
                state.mode = "cached"
            state.slot, state.slot_key = slot, key
            record["reason"] = reason

            ref = None
            if state.mode == "cached" and state.verify:
                mode, state.mode = state.mode, "off"
                ref = executor(x, timestep, context, transformer_options, **kwargs)
                ref = [t.detach().clone() for t in ref]
                state.mode = mode
                state.counts = {"build": 0, "cached": 0, "stock": 0}

            t0 = time.perf_counter()
            try:
                out = executor(x, timestep, context, transformer_options, **kwargs)
            except BaseException:
                if state.mode == "build":
                    state.slots.pop(key, None)
                raise
            finally:
                slot.dq = None
            if x[0].device.type == "cuda":
                torch.cuda.synchronize(x[0].device)
            record["mode"] = state.mode
            record["counts"] = dict(state.counts)
            record["seconds"] = time.perf_counter() - t0
            if state.mode == "build":
                slot.complete = all(e is not None for e in slot.h)
                record["cache_bytes"] = slot.nbytes()
                log.info("[h3] frozen video cache: built (%s), %d rows, %s, %.2f GiB in RAM",
                         reason, layout.seq_len, slot.codec.name, slot.nbytes() / 2**30)
            if ref is not None:
                a, b = out[1].float().flatten(), ref[1].float().flatten()
                cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
                rel = float((a - b).norm() / b.norm().clamp(min=1e-12))
                record["verify"] = {"audio_cos": cos, "audio_rel_l2": rel}
                state.verify_log.append(record["verify"])
                log.info("[h3] frozen video cache: verify, sigma %.4f, audio velocity "
                         "cosine %.6f, relative L2 %.4g against the stock step",
                         1.0 - t_v, cos, rel)
            state.calls.append(record)
            return out
        finally:
            state.mode = "off"
            state.slot = None
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


def attach(model, precision="int4", refresh=False, refresh_every=2, verify=False):
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
            f"core's sparse attention, taomate or FunControl. A cached step would skip "
            f"them, so the cache refuses to compose. Put it on a model without them.")
    m = model.clone()
    state = _State(dm, precision, refresh, refresh_every, verify)
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
                "For an audio-only refine pass (MiniMax H3 Audio Refine Mask): the first "
                "step runs stock and keeps each block's attention input in RAM; later "
                "steps compute only the text and audio rows against it. Active only on "
                "calls whose video is frozen and whose audio is open; every other call "
                "runs stock. The cached rows stop reacting to the new audio, which "
                "`verify` measures. Ported from ComfyUI-H3-AudioRefine (MIT)."),
            inputs=[
                io.Model.Input("model", tooltip="The refine pass's model, before any distill LoRA."),
                io.Combo.Input("precision", options=list(CODECS), default="int4",
                               tooltip="How the kept hidden states are stored in RAM. int4 is the "
                                       "smallest; bf16 is exact to the model's compute dtype."),
                io.Boolean.Input("refresh", default=False, advanced=True,
                                 tooltip="Rebuild the cache every `refresh_every` cached steps."),
                io.Int.Input("refresh_every", default=2, min=1, max=100, advanced=True,
                             tooltip="Cached steps between rebuilds, when `refresh` is on."),
                io.Boolean.Input("verify", default=False, advanced=True,
                                 tooltip="Also run each cached step stock and log the audio "
                                         "velocity's cosine against it. Costs a full step each."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, precision="int4", refresh=False, refresh_every=2,
                verify=False) -> io.NodeOutput:
        return io.NodeOutput(attach(model, precision, refresh, refresh_every, verify))
