"""A bench-only ComfyUI node: can the encoder's work on a reference be kept
across prompt edits without changing the conditioning?

NOT part of the pack. A server sees it only when launched with an
`--extra-model-paths-config` whose `custom_nodes` entry names
`bench/comfy_capture_nodes/`, which `bench/measure_encoder_prefix_reuse.py`
writes and prints. It runs inside the server because the shipped encoder does
not fit the card without core's dynamic VRAM, which a bare process does not
have (tried 2026-10-03: core's own encode runs out of memory there). The
record is written under `H3_BENCH_CAPTURE_DIR`, read from the server's
environment so no path is typed into a graph.

## Why it should be possible

Core presents an H3 reference to Qwen3-VL ahead of the prompt
(`comfy/text_encoders/minimax.py::MiniMaxH3Tokenizer`) and the language
model's attention is causal (`comfy/text_encoders/llama.py::Llama2_.forward`).
So what the encoder computes on the reference prefix cannot depend on the
prompt: run the prefix once, keep its output rows and each layer's keys and
values, then run only the prompt tokens against them.

## The cases, each a `torch.equal` on conditioning rows

  repeat              core's full pass twice on one prompt. The floor: if
                      this is not exact, nothing below can be.
  replica             this file's layer loop on the full sequence, against
                      core's pass. Core's forward cannot continue more than a
                      few tokens from a cache (its causal mask is square over
                      past plus new), so the loop below is a copy of core's,
                      and has to be shown to be core's arithmetic first.
  prefix_two_prompts  core's full pass on the prompt and on an edit of it:
                      the prefix rows. Causality as core computes it, the
                      injected visual features included.
  prefix_short_prompt the same against a prompt cut to a quarter: whether
                      today's own prefix rows are stable when the sequence
                      length moves by more than a few tokens. If they are
                      not, "the same as today" is itself only defined up to
                      that difference.
  prefix_alone        the loop on the prefix only, against those rows of the
                      full pass: a shorter sequence through the same kernels.
  continued           the loop on the prompt tokens only, over the prefix's
                      cached keys and values, against those rows of the full
                      pass. This is the reuse.
  continued_edit      the same for the edited prompt, on the first prompt's
                      cache.
  prompt_without_image  NOT a reuse: core's encode of the prompt as plain
                      text, no reference in context, against the prompt rows
                      of the joint pass. What splicing separately encoded
                      reference rows onto a text-only conditioning gets wrong
                      on the prompt's side, for scale beside `continued`.
"""

import json
import os
import sys
import time

import torch


def _pack_module(suffix):
    """The H3 pack's module as this server loaded it, whatever name the
    custom-node loader gave the package."""
    for name, module in list(sys.modules.items()):
        if name.endswith("." + suffix) and "h3" in (getattr(module, "__file__", "") or "").lower():
            return module
    raise RuntimeError(f"the H3 pack's {suffix} is not loaded in this server")


def compare(a, b):
    """torch.equal, and how far apart when not."""
    a, b = a.float().cpu(), b.float().cpu()
    row = {"rows": int(a.shape[1]), "equal": bool(a.shape == b.shape and torch.equal(a, b))}
    if not row["equal"] and a.shape == b.shape:
        d = a - b
        row["rel_l2"] = float(d.norm() / a.norm())
        row["max_abs"] = float(d.abs().max())
        row["rows_differing"] = int((d.abs().amax(dim=-1) > 0).sum())
        row["worst_row_rel_l2"] = float((d.norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-12)).max())
    return row


def measure(clip, ref_items, prompt, edit, normalize):
    import comfy.model_prefetch
    from comfy.ldm.modules.attention import optimized_attention_for_device

    te = clip.cond_stage_model
    sdclip = getattr(te, te.clip)
    lm = sdclip.transformer.model

    def tokens_for(text):
        return clip.tokenize(normalize(text), minimax_ref_items=ref_items)

    def timed(fn, *a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    def core_pass(text):
        return clip.encode_from_tokens(tokens_for(text))

    def prepared(text):
        """What core hands the language model for this prompt: the embedded
        sequence (vision tower run), positions, and the visual features it
        injects in the first layers. As `comfy.sd.CLIP.encode_from_tokens`
        and `SDClipModel.forward` reach it."""
        tokens = tokens_for(text)
        te.reset_clip_options()
        clip.load_model(tokens)
        device = clip.patcher.load_device
        te.set_clip_options({"execution_device": device})
        entries = [pair[0] for pair in tokens[te.clip_name][0]]
        embeds, _mask, _n, info = sdclip.process_tokens([entries], device)
        position_ids, visual, deepstack = sdclip.transformer.build_image_inputs(embeds, info)
        images = [e for e in info if e.get("type") == "image"]
        prefix = max(e["index"] + e["size"] for e in images) + 1      # past the closing vision token
        return embeds, position_ids, visual, deepstack, prefix

    def lm_pass(x, position_ids, visual=None, deepstack=None, past=None):
        """`Llama2_.forward`'s layer loop for the conditioning path, with one
        difference: the causal mask keeps only the new tokens' rows, so a
        cache of earlier keys and values can sit under more than one token.
        Returns (hidden states, per-layer (keys, values, length))."""
        x = x.clone()                       # a layer writes its output into its input
        seq, past_len = x.shape[1], (past[0][2] if past else 0)
        freqs_cis = lm.compute_freqs_cis(position_ids, x.device)
        total = past_len + seq
        mask = None
        if total > 1:
            mask = torch.empty(total, total, dtype=x.dtype, device=x.device).fill_(
                torch.finfo(x.dtype).min / 4).triu_(1)[past_len:]
        attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)
        queue = comfy.model_prefetch.make_prefetch_queue(
            list(lm.layers), x.device, {"prefetch_dynamic_vbars": False})
        cache = []
        for i, layer in enumerate(lm.layers):
            def core():
                nonlocal x
                x, kv = layer(x=x, attention_mask=mask, freqs_cis=freqs_cis,
                              optimized_attention=attention,
                              past_key_value=past[i] if past else [])
                cache.append(kv)
            comfy.model_prefetch.prefetch_queue_pop(
                queue, x.device, layer, x.dtype, core=core, enable_graph=False, malloc_scope="block")
            if deepstack is not None and i < len(deepstack):
                x[visual] = x[visual] + deepstack[i].to(x)
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="block")
        if lm.norm is not None:
            x = lm.norm(x)
        return x, cache

    cases, seconds = {}, {}
    full, seconds["core_full_first"] = timed(core_pass, prompt)
    again, seconds["core_full"] = timed(core_pass, prompt)
    cases["repeat"] = compare(full, again)
    del again
    full_edit, seconds["core_full_edit"] = timed(core_pass, edit)

    embeds, position_ids, visual, deepstack, prefix = prepared(prompt)
    tokens_total = int(embeds.shape[1])
    (mine, _), seconds["loop_full"] = timed(lm_pass, embeds, position_ids, visual, deepstack)
    cases["replica"] = compare(full, mine)
    del mine
    cases["prefix_two_prompts"] = compare(full[:, :prefix], full_edit[:, :prefix])
    short, seconds["core_full_short_prompt"] = timed(core_pass, prompt[: max(1, len(prompt) // 4)])
    cases["prefix_short_prompt"] = compare(full[:, :prefix], short[:, :prefix])
    short_tokens = int(short.shape[1])
    del short

    (head, cache), seconds["loop_prefix"] = timed(
        lm_pass, embeds[:, :prefix], position_ids[:, :prefix], visual[:, :prefix], deepstack)
    cases["prefix_alone"] = compare(full[:, :prefix], head)
    del head
    cache_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v, _n in cache)
    (tail, _), seconds["loop_continued"] = timed(
        lm_pass, embeds[:, prefix:], position_ids[:, prefix:], past=cache)
    cases["continued"] = compare(full[:, prefix:], tail)
    del tail

    embeds2, position_ids2, _v, _d, prefix2 = prepared(edit)
    if prefix2 != prefix:
        raise RuntimeError(f"the edit moved the prefix: {prefix} against {prefix2}")
    cases["embeds_prefix_two_prompts"] = compare(embeds[:, :prefix], embeds2[:, :prefix])
    (tail2, _), seconds["loop_continued_edit"] = timed(
        lm_pass, embeds2[:, prefix:], position_ids2[:, prefix:], past=cache)
    cases["continued_edit"] = compare(full_edit[:, prefix:], tail2)

    plain = clip.encode_from_tokens(clip.tokenize(normalize(prompt)))
    cases["prompt_without_image"] = compare(full[:, prefix:], plain)
    if not cases["prompt_without_image"]["equal"] and plain.shape[1] == tokens_total - prefix:
        a, b = full[:, prefix:].float().cpu(), plain.float().cpu()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        cases["prompt_without_image"]["row_cos_min"] = float(cos.min())
        cases["prompt_without_image"]["row_cos_median"] = float(cos.median())
    cases["prompt_without_image"]["plain_rows"] = int(plain.shape[1])

    return {
        "tokens": {"total": tokens_total, "prefix": int(prefix), "prompt": tokens_total - int(prefix),
                   "total_edit": int(full_edit.shape[1]), "total_short_prompt": short_tokens},
        "hidden_dtype": str(embeds.dtype),
        "cache": {"bytes": int(cache_bytes), "dtype": str(cache[0][0].dtype), "layers": len(cache),
                  "keys_shape": list(cache[0][0].shape)},
        "attention": getattr(optimized_attention_for_device(embeds.device, mask=True, small_input=True),
                             "__name__", "?"),
        "cases": cases,
        "seconds": {k: round(v, 2) for k, v in seconds.items()},
        "seconds_note": "core_full* include tokenizing and the vision tower; the loop_* passes are the "
                        "language model alone",
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
    }


class H3BenchEncoderPrefixReuse:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",),
                             "references": ("MINIMAX_H3_REFERENCES",),
                             "prompt": ("STRING", {"multiline": True}),
                             "edit": ("STRING", {"multiline": True}),
                             "name": ("STRING", {"default": "prefix_reuse"})}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "h3/bench"

    def run(self, clip, references, prompt, edit, name):
        out = os.environ.get("H3_BENCH_CAPTURE_DIR")
        if not out:
            raise RuntimeError("H3_BENCH_CAPTURE_DIR is not set in the server's environment")
        os.makedirs(out, exist_ok=True)
        rc = _pack_module("reference_conditioning")
        # No VAE: the encoder's presentation alone, which is what is under test.
        ref_items, _blocks = rc._compile_reference_records(tuple(references), None, None, 1344, 768, 345)
        record = measure(clip, ref_items, prompt, edit, rc.normalize_prompt)
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump(record, f, indent=1)
        return {"ui": {"text": (json.dumps(record["cases"]),)}}


def _joint(clip, ref_items, text):
    """Core's one-pass conditioning, as MiniMaxH3ReferenceConditioning makes it."""
    return clip.encode_from_tokens_scheduled(clip.tokenize(text, minimax_ref_items=ref_items))


def compare_conditioning(joint, split, encoded):
    """The split path's conditioning against core's: rows per reference span
    and for the prompt, the token tags, and the shape of the extras."""
    a, b = joint[0][0], split[0][0]
    out = {"shape_equal": tuple(a.shape) == tuple(b.shape), "spans": []}
    if not out["shape_equal"]:
        out["shapes"] = [list(a.shape), list(b.shape)]
        return out
    start = 0
    for span in encoded.spans:
        out["spans"].append(compare(a[:, start:start + span.length], b[:, start:start + span.length]))
        start += span.length
    out["prompt"] = compare(a[:, start:], b[:, start:])
    ea, eb = joint[0][1], split[0][1]
    out["extra_keys_equal"] = sorted(ea) == sorted(eb)
    out["extra_keys"] = [sorted(ea), sorted(eb)]
    ta, tb = ea.get("minimax_token_tags"), eb.get("minimax_token_tags")
    out["tags_equal"] = bool(ta is not None and tb is not None and ta.shape == tb.shape
                             and torch.equal(ta.cpu(), tb.cpu()))
    pa, pb = ea.get("pooled_output"), eb.get("pooled_output")
    out["pooled_equal"] = (pa is None and pb is None) or (
        pa is not None and pb is not None and bool(torch.equal(pa.cpu(), pb.cpu())))
    out["dtype_device"] = [str(a.dtype), str(a.device), str(b.dtype), str(b.device)]
    return out


class H3BenchSplitReferences:
    """The pack's two-node path (`reference_encode.py`) against core's one
    pass, on the real encoder: one reference, the same encoded again, then two
    chains that share their first reference. Nothing is kept between encodes
    (the pack's reference store was removed on 2026-10-10), so "again" is a
    second encode whose rows must equal the first's."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",),
                             "references_a": ("MINIMAX_H3_REFERENCES",),
                             "references_ab": ("MINIMAX_H3_REFERENCES",),
                             "references_ac": ("MINIMAX_H3_REFERENCES",),
                             "prompt": ("STRING", {"multiline": True}),
                             "edit": ("STRING", {"multiline": True}),
                             "name": ("STRING", {"default": "split_references"})}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "h3/bench"

    def run(self, clip, references_a, references_ab, references_ac, prompt, edit, name):
        out = os.environ.get("H3_BENCH_CAPTURE_DIR")
        if not out:
            raise RuntimeError("H3_BENCH_CAPTURE_DIR is not set in the server's environment")
        os.makedirs(out, exist_ok=True)
        rc = _pack_module("reference_conditioning")
        enc = _pack_module("reference_encode")
        prompt, edit = rc.normalize_prompt(prompt), rc.normalize_prompt(edit)

        def items(references):
            return rc._compile_reference_records(tuple(references), None, None, 1344, 768, 345)[0]

        def timed(fn, *a):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            value = fn(*a)
            torch.cuda.synchronize()
            return value, round(time.perf_counter() - t0, 2)

        record = {"cases": {}, "seconds": {}, "reports": {}}
        cases, seconds, reports = record["cases"], record["seconds"], record["reports"]

        a = items(references_a)
        joint_a, seconds["joint_a"] = timed(_joint, clip, a, prompt)
        encoded_a, seconds["encode_a"] = timed(enc.encode_references, clip, a)
        reports["encode_a"] = encoded_a.report
        split_a, seconds["prompt_on_a"] = timed(enc.condition_prompt, encoded_a, prompt)
        cases["one_reference"] = compare_conditioning(joint_a, split_a, encoded_a)

        joint_a_edit, seconds["joint_a_edit"] = timed(_joint, clip, a, edit)
        again, seconds["encode_a_again"] = timed(enc.encode_references, clip, a)
        reports["encode_a_again"] = again.report
        cases["encoded_again_same_rows"] = all(bool(torch.equal(x.rows, y.rows)) for x, y in zip(again.spans, encoded_a.spans))
        split_a_edit, seconds["prompt_edit_on_a"] = timed(enc.condition_prompt, again, edit)
        cases["one_reference_edit"] = compare_conditioning(joint_a_edit, split_a_edit, again)
        repeat, _ = timed(enc.condition_prompt, again, edit)
        cases["split_repeat_equal"] = bool(torch.equal(repeat[0][0], split_a_edit[0][0]))
        del joint_a, split_a, joint_a_edit, split_a_edit, repeat

        for label, references in (("ab", references_ab), ("ac", references_ac), ("ab_again", references_ab)):
            both = items(references)
            encoded, seconds[f"encode_{label}"] = timed(enc.encode_references, clip, both)
            reports[f"encode_{label}"] = encoded.report
            split, seconds[f"prompt_on_{label}"] = timed(enc.condition_prompt, encoded, prompt)
            if label != "ab_again":
                joint, seconds[f"joint_{label}"] = timed(_joint, clip, both, prompt)
                cases[f"chain_{label}"] = compare_conditioning(joint, split, encoded)
                cases[f"chain_{label}"]["first_span_rows_are_a"] = bool(torch.equal(encoded.spans[0].rows, encoded_a.spans[0].rows))
                del joint
            record.setdefault("tokens", {})[label] = [s.length for s in encoded.spans]
            record.setdefault("span_bytes", {})[label] = [s.nbytes for s in encoded.spans]
            del split
        record["tokens"]["a"] = [s.length for s in encoded_a.spans]
        record["torch"], record["device"] = torch.__version__, torch.cuda.get_device_name(0)
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump(record, f, indent=1)
        return {"ui": {"text": (json.dumps(record["cases"]),)}}


NODE_CLASS_MAPPINGS = {"H3BenchEncoderPrefixReuse": H3BenchEncoderPrefixReuse,
                       "H3BenchSplitReferences": H3BenchSplitReferences}
