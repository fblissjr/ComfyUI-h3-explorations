"""Encode an H3 reference list once, and write prompts on top of it.

`MiniMaxH3ReferenceConditioning` (`reference_conditioning.py`) presents the
references and the prompt to Qwen3-VL as one sequence and encodes it whole, so
a prompt edit pays for the references again. Nearly all of that is the encoder
reading the stills (`data/results/2026-10-03_prompt_edit_conditioning_cost.json`).

Core places every reference ahead of the prompt
(`comfy/text_encoders/minimax.py::MiniMaxH3Tokenizer`) and the language
model's attention is causal (`comfy/text_encoders/llama.py::Llama2_.forward`),
so what the encoder computes for the references cannot depend on the prompt.
The two nodes here split the work along that line:

  MiniMaxH3EncodeReferences   the references alone: the vision tower, the
                              language model over each reference's span, the
                              VAE rows. Its inputs do not include the prompt,
                              so ComfyUI's node cache keeps its output across
                              prompt edits.
  MiniMaxH3PromptOnReferences the prompt tokens alone, attending to the
                              references' kept keys and values.

## What is the same as the one-node path, and what is not

Measured on the shipped encoder with one 2048 still
(`bench/results/2026-10-03_encoder_prefix_reuse.json`): the references' rows
are the same bytes as the one-node path's at that shape; the prompt's rows
differ in the last bits (relative difference of a few millionths, where the
int8 encoder sits a few thousandths from bf16). The owner accepted that
difference on 2026-10-03. At other shapes the references' rows may also differ
in the last bits. Nothing here is the default: every shipped graph still wires
the one-node path.

## Nothing is kept between runs

Until 2026-10-10 each reference's span was kept in host memory under a
chained hash of what the language model was handed for it, and handed back
on a later run when the hash matched (and `keep_references` on the one-node
path reached that store without rewiring). The owner's rule of that day
removed it: nothing an earlier run made is reused by a key; a result is made
again, or handed in explicitly as a named file or a wired input
(`AGENTS.md`, "Operative rules"; `docs/wiki/decisions.md`). What is left is
the explicit form. `MiniMaxH3EncodeReferences` hands its result out on an
output and `MiniMaxH3PromptOnReferences` takes it on an input, so a prompt
edit that leaves the references' node untouched pays for the prompt alone,
through ComfyUI's own node cache and nothing of this pack's.

Within one encode a reference still continues on the references before it:
a span depends on everything ahead of it and on nothing after it. Nothing
kept stays on the card: a pass hands each layer its keys and values as the
loop reaches it and drops them after.

## The loop below is a copy of core's

`Llama2_.forward` cannot continue more than a few tokens from kept keys and
values: its causal mask is square over past plus new, and its list path
collects no keys when handed an empty list. `_lm_pass` is that loop with the
mask cut to the new tokens' rows. `bench/check_reference_encode.py` pins the
source of the core functions it copies, and the bench node
`bench/comfy_capture_nodes/h3_bench_prefix_reuse` holds it to core's output on
the real encoder.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import node_helpers
import torch
from comfy_api.latest import io, ui
from comfy_extras.nodes_minimax_h3 import _empty_av_latent

from .h3_rules import normalize_prompt
from .reference_conditioning import (
    AUDIO_VAE_TOOLTIP,
    CLIP_TOOLTIP,
    HEIGHT_TOOLTIP,
    IMAGE_POLICIES,
    IMAGE_POLICY_TOOLTIP,
    LENGTH_TOOLTIP,
    PROMPT_TOOLTIP,
    REFERENCES_LIST_TOOLTIP,
    VIDEO_POLICIES,
    VIDEO_POLICY_TOOLTIP,
    VIDEO_VAE_TOOLTIP,
    WIDTH_TOOLTIP,
    H3References,
    _compile_reference_records,
    _order_records,
    _reference_tuple,
)
from .reference_order import assign_labels

logger = logging.getLogger(__name__)

H3EncodedReferences = io.Custom("MINIMAX_H3_ENCODED_REFERENCES")

@dataclass
class Span:
    """One reference's stretch of the encoder's sequence."""

    length: int
    rows: torch.Tensor                      # [1, length, hidden], float32, host
    keys: list[torch.Tensor]                # per layer [1, kv heads, length, head dim], host
    values: list[torch.Tensor]
    images: list[dict]                      # vision blocks, indexed from this span's start
    nbytes: int = 0

    def __post_init__(self):
        self.nbytes = (self.rows.numel() * self.rows.element_size()
                       + sum(t.numel() * t.element_size() for t in self.keys)
                       + sum(t.numel() * t.element_size() for t in self.values))


@dataclass
class EncodedReferences:
    """What `MiniMaxH3EncodeReferences` hands on. Holds its spans itself, so
    it stays whole if the store evicts them."""

    spans: list[Span]
    ref_blocks: list[dict]
    ref_items: list[dict]
    clip: Any                               # the CLIP that encoded them; the prompt node uses it
    patches: Any
    labels: list[str] | None = None
    report: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return sum(s.length for s in self.spans)

    def get_models(self):
        """For core's `PromptModelTracker`: the encoder is in use by a prompt
        that uses these references, also on a run where the encode node is
        served from cache and only the prompt node executes."""
        return [self.clip.patcher]

    def _comfy_cache_tensors(self):
        """For core's RAM-pressure cache, which sizes an entry by its tensors."""
        out = []
        for span in self.spans:
            out += [span.rows, *span.keys, *span.values]
        return out

    def images(self) -> list[dict]:
        """Every vision block, indexed from the start of the sequence, in the
        shape core's position and tag builders read."""
        out, start = [], 0
        for span in self.spans:
            out += [dict(e, index=e["index"] + start) for e in span.images]
            start += span.length
        return out



def _kept_layer(spans, i: int, device):
    """Layer `i`'s kept keys and values for a chain of spans, on the device."""
    if len(spans) == 1:
        return spans[0].keys[i].to(device), spans[0].values[i].to(device)
    return (torch.cat([s.keys[i].to(device) for s in spans], dim=2),
            torch.cat([s.values[i].to(device) for s in spans], dim=2))


def _parts(clip):
    """(text encoder model, its Qwen3-VL clip model, the language model), or a
    refusal when this is not core's H3 encoder."""
    te = clip.cond_stage_model
    sdclip = getattr(te, getattr(te, "clip", ""), None)
    lm = getattr(getattr(sdclip, "transformer", None), "model", None)
    if getattr(te, "clip_name", None) != "qwen3vl_32b" or lm is None or not hasattr(lm, "layers"):
        raise ValueError(
            "these nodes take the MiniMax H3 text encoder (Qwen3-VL); this CLIP is "
            f"{type(te).__name__}. Wire MiniMax H3 Encoder Loader's output.")
    return te, sdclip, lm


def unsupported(clip):
    """Why this CLIP cannot be split, or None. A clip-skip or scheduled hooks
    change what core's own encode does in ways the copied loop does not
    reproduce."""
    if clip.layer_idx is not None:
        return "a clip layer is selected on this encoder"
    if clip.patcher.forced_hooks is not None and clip.use_clip_schedule:
        return "this encoder carries scheduled hooks"
    return None


def span_groups(ref_items):
    """Which presentation items share a span, as lists of item indices.

    One span per item that carries vision tokens. An item that is only a label
    (an audio reference, a video's soundtrack) joins the item after it, since
    a language-model pass has a fixed cost and a few label tokens do not earn
    one; trailing label-only items form a last span. The rule reads the list
    and nothing else, so a chain is cut the same way whatever is in the store,
    and a chain's kept values do not depend on the order things were encoded."""
    groups, pending = [], []
    for i, item in enumerate(ref_items):
        pending.append(i)
        if item.get("type") != "audio":
            groups.append(pending)
            pending = []
    if pending:
        groups.append(pending)
    return groups


def span_entries(clip, ref_items, groups=None):
    """The tokenizer's entries for the reference list, one list per span.

    Core appends each reference's label and vision blocks in order and numbers
    labels by what came before, so the entries for the first i references are
    a prefix of the entries for the first i + 1, and the cut points are the
    lengths."""
    name = clip.cond_stage_model.clip_name
    groups = span_groups(ref_items) if groups is None else groups
    cuts, entries = [0], []
    for group in groups:
        entries = clip.tokenize("", minimax_ref_items=ref_items[:group[-1] + 1])[name][0]
        cuts.append(len(entries))
    return [entries[a:b] for a, b in zip(cuts, cuts[1:])]


def _positions(images, start: int, count: int, device):
    """Core's position ids for `count` tokens from `start`, given every vision
    block up to there: `qwen2vl_mrope_position_ids` when there is one, a plain
    count when there is none (what `Llama2_.forward` builds itself)."""
    from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids
    if not images:
        return torch.arange(start, start + count, device=device).unsqueeze(0)
    return qwen2vl_mrope_position_ids(images, start + count, device)[:, start:]


def _causal_rows(seq: int, past_len: int, dtype, device):
    """The new tokens' rows of core's causal mask over past plus new: rows
    `past_len` onward of `Llama2_.forward`'s `triu_(1)` square, built without
    the square."""
    return torch.full((seq, past_len + seq), torch.finfo(dtype).min / 4,
                      dtype=dtype, device=device).triu_(past_len + 1)


def _lm_pass(lm, x, position_ids, visual=None, deepstack=None, past=None, past_len=0,
             keep=True):
    """`Llama2_.forward`'s layer loop for the conditioning path, continued on
    kept keys and values.

    `past(i)` returns layer i's kept (keys, values) on x's device, covering
    `past_len` tokens; None for a pass from the start. Two differences from
    core's loop, neither in the arithmetic: the causal mask holds only the new
    tokens' rows (core's is square over past plus new, which is why its own
    forward cannot continue more than a few tokens), and each layer's kept
    pair is fetched when the loop reaches it and dropped after.

    Returns (hidden states, per-layer (keys, values) of the NEW tokens on the
    host) with `keep`, else (hidden states, None)."""
    import comfy.model_management as mm
    import comfy.model_prefetch
    from comfy.ldm.modules.attention import optimized_attention_for_device

    # A layer writes its output into its input, so `x` is consumed. Both
    # callers hand over embeddings nothing else reads.
    seq = x.shape[1]
    freqs_cis = lm.compute_freqs_cis(position_ids, x.device)
    total = past_len + seq
    mask = _causal_rows(seq, past_len, x.dtype, x.device) if total > 1 else None
    attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)
    # Core passes `prefetch_dynamic_vbars` only when it is generating from a
    # cache; the conditioning pass runs without it, and so does this, so the
    # continued pass stays on the path that was measured against core.
    queue = comfy.model_prefetch.make_prefetch_queue(
        list(lm.layers), x.device, {"prefetch_dynamic_vbars": False})
    host = mm.intermediate_device()
    new = [] if keep else None
    for i, layer in enumerate(lm.layers):
        mm.throw_exception_if_processing_interrupted()

        def core():
            nonlocal x
            kept = [] if past is None else (*past(i), past_len)
            x, kv = layer(x=x, attention_mask=mask, freqs_cis=freqs_cis,
                          optimized_attention=attention, past_key_value=kept)
            if keep:
                # copy=True: when the encoder runs on the host, `.to` alone would
                # return a view that keeps the whole past-plus-new tensor alive.
                new.append((kv[0][:, :, past_len:].to(host, copy=True),
                            kv[1][:, :, past_len:].to(host, copy=True)))
        comfy.model_prefetch.prefetch_queue_pop(
            queue, x.device, layer, x.dtype, core=core, enable_graph=False, malloc_scope="block")
        if deepstack is not None and i < len(deepstack):
            x[visual] = x[visual] + deepstack[i].to(x)
    comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="block")
    if lm.norm is not None:
        x = lm.norm(x)
    return x, new


class _Loaded:
    """The encoder on its device for a pass, as `comfy.sd.CLIP.encode_from_tokens`
    prepares it."""

    def __init__(self, clip, tokens):
        self.clip, self.tokens = clip, tokens

    def __enter__(self):
        import comfy.model_management as mm
        clip, te = self.clip, self.clip.cond_stage_model
        te.reset_clip_options()
        memory = 0
        if hasattr(te, "memory_estimation_function"):
            memory = te.memory_estimation_function(self.tokens, device=clip.patcher.load_device)
        mm.load_models_gpu([clip.patcher], memory_required=memory)
        device = clip.patcher.load_device
        te.set_clip_options({"execution_device": device})
        self.context = mm.cuda_device_context(device)
        self.context.__enter__()
        return device

    def __exit__(self, *exc):
        return self.context.__exit__(*exc)


def _encode_span(sdclip, lm, entries, start, images_before, before, device):
    """One reference's span, continued on the spans `before` it."""
    import comfy.model_management as mm
    embeds, _mask, _n, info = sdclip.process_tokens([[e[0] for e in entries]], device)
    count = int(embeds.shape[1])
    local = sorted((e for e in info if e.get("type") == "image"), key=lambda e: e["index"])
    images = [{"type": "image", "index": int(e["index"]), "size": int(e["size"]),
               "extra": {"grid": e["extra"]["grid"].cpu()}} for e in local]
    everything = images_before + [dict(e, index=e["index"] + start) for e in images]
    position_ids = _positions(everything, start, count, device)
    visual = deepstack = None
    if local:
        # As `Qwen3VL.build_image_inputs`: where the vision tokens sit, and the
        # per-layer features core adds there in the first layers.
        visual = torch.zeros((1, count), dtype=torch.bool, device=device)
        for e in local:
            visual[0, e["index"]:e["index"] + e["size"]] = True
            ds = e["extra"]["deepstack"]
            deepstack = list(ds) if deepstack is None else [
                torch.cat([deepstack[i], ds[i]], dim=0) for i in range(len(ds))]
    past = (lambda i: _kept_layer(before, i, device)) if before else None
    x, new = _lm_pass(lm, embeds, position_ids, visual, deepstack, past, start)
    return Span(length=count, rows=x.float().to(mm.intermediate_device()),
                keys=[k for k, _v in new], values=[v for _k, v in new], images=images)


def _described(ref_items, labels):
    """One name per presentation item for the report: its label, and a still's
    size as the encoder sees it."""
    out = []
    for i, item in enumerate(ref_items):
        name = labels[i] if labels and i < len(labels) else f"reference {i + 1}"
        data = item.get("data")
        if item.get("type") == "image" and torch.is_tensor(data) and data.ndim == 4:
            name += f" ({data.shape[2]}x{data.shape[1]} to the encoder)"
        out.append(name)
    return out


def encode_references(clip, ref_items, labels=None) -> EncodedReferences:
    """Each reference's span, encoded now, each continued on the ones before it.
    `labels` are the prompt's names for the items, in order, for the report."""
    reason = unsupported(clip)
    if reason is not None:
        raise ValueError(
            f"this encoder's pass cannot be split: {reason}. Use MiniMax H3 Reference "
            "Conditioning, which encodes in one pass.")
    te, sdclip, lm = _parts(clip)
    patches = clip.patcher.patches_uuid
    groups = span_groups(ref_items)
    per_reference = span_entries(clip, ref_items, groups)
    described = _described(ref_items, labels)
    names = [" and ".join(described[i] for i in group) for group in groups]
    spans, report = [], []
    with _Loaded(clip, clip.tokenize("", minimax_ref_items=ref_items)) as device:
        start, images = 0, []
        for i, entries in enumerate(per_reference):
            t0 = time.perf_counter()
            how = "encoded" if not spans else "continued on the references before it"
            span = _encode_span(sdclip, lm, entries, start, images, list(spans), device)
            report.append(f"{names[i]}: {how} ({span.length} tokens, {time.perf_counter() - t0:.1f} s)")
            spans.append(span)
            images += [dict(e, index=e["index"] + start) for e in span.images]
            start += span.length
    chain = sum(s.nbytes for s in spans)
    report.append(f"this encode holds {chain / 2 ** 30:.1f} GiB on the host for as long as its output is wired "
                  "(nothing is kept between runs, on the card or on disk)")
    for line in report:
        logger.info("[h3-ref] %s", line)
    return EncodedReferences(spans=list(spans), ref_blocks=[], ref_items=list(ref_items),
                             clip=clip, patches=patches,
                             labels=list(labels) if labels else None, report=report)


def condition_prompt(encoded: EncodedReferences, text: str):
    """The conditioning list for `text` on top of `encoded`, in the shape
    `CLIP.encode_from_tokens_scheduled` returns. Runs on the CLIP that encoded
    the references, which they carry."""
    import comfy.model_management as mm
    from comfy.text_encoders.minimax import token_tags_from_embeds_info

    clip = encoded.clip
    if unsupported(clip) is not None or encoded.patches != clip.patcher.patches_uuid:
        # The encoder changed under the kept references (a layer selected on
        # it, new patches). They hold what they were made from, so make them
        # again; `encode_references` refuses what cannot be split.
        logger.info("[h3-ref] the encoder changed since these references were encoded; encoding them again")
        fresh = encode_references(clip, encoded.ref_items, encoded.labels)
        fresh.ref_blocks = encoded.ref_blocks
        encoded = fresh
    te, sdclip, lm = _parts(clip)
    tokens = clip.tokenize(text)
    start, spans = encoded.length, encoded.spans
    with _Loaded(clip, tokens) as device:
        entries = [pair[0] for pair in tokens[te.clip_name][0]]
        embeds, _mask, _n, _info = sdclip.process_tokens([entries], device)
        count = int(embeds.shape[1])
        images = encoded.images()
        position_ids = _positions(images, start, count, device)
        past = (lambda i: _kept_layer(spans, i, device)) if spans else None
        x, _new = _lm_pass(lm, embeds, position_ids, past=past, past_len=start, keep=False)
        rows = x.float().to(mm.intermediate_device())
    cond = torch.cat([s.rows for s in spans] + [rows], dim=1)
    extra = {"pooled_output": None,
             "minimax_token_tags": token_tags_from_embeds_info(start + count, images)}
    clip.add_hooks_to_dict(extra)
    return [[cond, extra]]


class MiniMaxH3EncodeReferences(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EncodeReferences",
            display_name="MiniMax H3 Encode References",
            category="MiniMaxH3/references",
            is_experimental=True,
            description=(
                "Encode a reference chain once, separately from the prompt, and "
                "output an empty latent of the right size. Wire the result into "
                "MiniMax H3 Prompt On References. Editing the prompt then re-runs "
                "only that node, which is fast.\n\n"
                "Encoded references stay in memory for the session, so going "
                "back to a reference used a moment ago, or changing only the "
                "last one, reuses the rest. After a run, the preview says what "
                "was reused and what was encoded."
            ),
            # The tooltips are `reference_conditioning.py`'s own, one copy each:
            # the inputs mean the same thing on both nodes.
            inputs=[
                io.Clip.Input("clip", tooltip=CLIP_TOOLTIP),
                io.Vae.Input("vae", optional=True, tooltip=VIDEO_VAE_TOOLTIP),
                io.Vae.Input("audio_vae", optional=True, tooltip=AUDIO_VAE_TOOLTIP),
                H3References.Input("references", tooltip=REFERENCES_LIST_TOOLTIP),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32, tooltip=WIDTH_TOOLTIP),
                io.Int.Input("height", default=768, min=32, max=16384, step=32, tooltip=HEIGHT_TOOLTIP),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip=LENGTH_TOOLTIP),
                io.Combo.Input("video_policy", options=list(VIDEO_POLICIES), default="comfy",
                               optional=True, tooltip=VIDEO_POLICY_TOOLTIP),
                io.Combo.Input("image_policy", options=list(IMAGE_POLICIES), default="comfy",
                               optional=True, tooltip=IMAGE_POLICY_TOOLTIP),
            ],
            outputs=[
                H3EncodedReferences.Output(display_name="encoded references"),
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, references, width=1344, height=768, length=124,
                video_policy="comfy", image_policy="comfy", vae=None, audio_vae=None):
        records = _reference_tuple(references)
        if not records:
            raise ValueError("MiniMaxH3EncodeReferences needs at least one appended reference")
        latent, frame_count = _empty_av_latent(width, height, length)
        ref_items, ref_blocks = _compile_reference_records(
            records, vae, audio_vae, width, height, frame_count,
            video_policy=video_policy, image_policy=image_policy)
        encoded = encode_references(clip, ref_items, assign_labels(_order_records(records)))
        encoded.ref_blocks = ref_blocks
        return io.NodeOutput(encoded, latent, ui=ui.PreviewText("\n".join(encoded.report)))


class MiniMaxH3PromptOnReferences(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptOnReferences",
            display_name="MiniMax H3 Prompt On References",
            category="MiniMaxH3/references",
            is_experimental=True,
            # "To within rounding": the prompt's rows come from a second pass, so
            # they differ from the one-node result in the last bits, not more
            # (`data/results/2026-10-03_reference_split.json`).
            description=(
                "Encode the prompt on top of references from MiniMax H3 Encode "
                "References, and output the conditioning for the sampler. There "
                "is no text encoder to wire: it uses the one that encoded the "
                "references. The result matches MiniMax H3 Reference "
                "Conditioning to within rounding."
            ),
            inputs=[
                H3EncodedReferences.Input(
                    "encoded_references", tooltip="From MiniMax H3 Encode References."),
                io.String.Input(
                    "prompt", multiline=True, dynamic_prompts=True,
                    tooltip=PROMPT_TOOLTIP),
            ],
            outputs=[io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def execute(cls, encoded_references, prompt):
        if not isinstance(encoded_references, EncodedReferences):
            raise TypeError("encoded_references must come from MiniMax H3 Encode References")
        if not prompt or not prompt.strip():
            raise ValueError(
                "MiniMaxH3PromptOnReferences needs a prompt; empty prompts condition "
                "on a pad token in core and are refused here")
        t0 = time.perf_counter()
        conditioning = condition_prompt(encoded_references, normalize_prompt(prompt))
        if encoded_references.ref_blocks:
            # Absent, not []: see MiniMaxH3ReferenceConditioning.
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"minimax_refs": encoded_references.ref_blocks})
        logger.info("[h3-ref] prompt on %d kept reference span(s), %d tokens ahead of it: %.1f s",
                    len(encoded_references.spans), encoded_references.length,
                    time.perf_counter() - t0)
        return io.NodeOutput(conditioning)
