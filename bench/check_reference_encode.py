#!/usr/bin/env python3
"""Hold the split reference path's pieces that need no encoder.

`reference_encode.py` encodes an H3 reference list apart from the prompt and
continues the prompt on kept keys and values, with a copy of core's language
model loop. The real comparison with core's one pass needs the encoder and is
`bench/measure_reference_split.py`. This file holds what can drift without
one:

  the copied loop's source   `_lm_pass` copies `Llama2_.forward`'s layer loop,
    is pinned                and relies on `Attention.forward`'s list path for
                             kept keys and on `TransformerBlock.forward`
                             writing into its input. Their source is hashed
                             against `PINNED`. A change in core fails here,
                             and the fix is to re-read the three, re-run the
                             server measurement, and move the pin: never to
                             move the pin alone.
  the loop against core      a small random `Llama2_` at the 32B config's
    on a small model         shape of things (deepstack, mrope, grouped key
                             heads): spans encoded one on another and a prompt
                             continued on them, against `Llama2_.forward` on
                             the whole sequence. Two vision spans; a
                             label-only span ahead of a vision span; a
                             one-token prompt; a still's span ahead of a span
                             shaped like a video reference, which core
                             tokenizes as several vision blocks with a
                             timestamp's text between them
                             (`MiniMaxH3Tokenizer.tokenize_with_weights`).
                             Not bit-equal at these sizes
                             (the last bits move with the shape, as on the
                             real encoder), so the bound is `SMALL_MODEL_TOL`
                             on the largest difference over the largest value.
                             RED CONTROL: positions off by one are far
                             outside it. From a peer session's review
                             (refdude, 2026-10-03).
  positions                  `_positions` against core's own
                             `qwen2vl_mrope_position_ids` on a layout with a
                             vision block in the first span and one in the
                             second, cut where the spans are cut, and a plain
                             count when no vision block precedes.
  the causal mask            `_causal_rows` is the new tokens' rows of the
                             square mask core builds, for a pass from the
                             start and for continued ones.
  the span cut               one span per item that carries vision tokens; a
                             label-only item joins the one after it; the cut
                             reads the list alone.
  nothing is kept            the module holds no store and no key function,
                             and the one-node path has no `keep_references`:
                             nothing an earlier run made is reused by a key
                             (2026-10-10; until then this item held the
                             store's budget, eviction and chained key).
  the report's labels        `reference_report.price_references` labels each
                             record with its own label when a sounded video
                             (two labels) comes first.
  the two nodes are          both are in the pack's node list, experimental,
    registered               and the prompt node takes the encode node's type.
                             RED CONTROL: off, neither function is called.

    PYTHONPATH=<ComfyUI root> python bench/check_reference_encode.py
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: sha256 of the source of the core functions `_lm_pass` copies or leans on,
#: read at ComfyUI e9027f2b on 2026-10-03, the day the copy was held to core's
#: output on the shipped encoder (bench/results/2026-10-03_encoder_prefix_reuse.json).
PINNED = {
    "Llama2_.forward": "164b0ad348cb0544",
    "Attention.forward": "e05700e116d6a4ce",
    "TransformerBlock.forward": "71b138c2606ad2a3",
}


#: Largest difference over largest value allowed between the split loop and
#: core's forward on the small model. Reasoned: about twenty times what the
#: review measured (5e-7), and a thousand times under the red control.
SMALL_MODEL_TOL = 1e-5


def small_model_cases(enc, check):
    """The pack's `_encode_span`, `_kept_layer` and `_lm_pass` against core's
    `Llama2_.forward`, on a random model small enough for the CPU."""
    import comfy.ops
    import torch
    from comfy.text_encoders.llama import Llama2_, Qwen3VL_32BConfig
    from comfy.text_encoders.qwen3vl import Qwen3VL

    torch.manual_seed(0)
    hidden, device = 64, torch.device("cpu")
    cfg = Qwen3VL_32BConfig(vocab_size=64, hidden_size=hidden, intermediate_size=96,
                            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2)
    lm = Llama2_(cfg, device="cpu", dtype=torch.float32, ops=comfy.ops.manual_cast).eval()
    for param in lm.parameters():
        if param.ndim > 1:
            torch.nn.init.normal_(param, std=0.2)
        else:
            torch.nn.init.ones_(param)

    class Embedded:                                    # stands in for the clip model's process_tokens
        def __init__(self, embeds, info):
            self.embeds, self.info = embeds, info

        def process_tokens(self, tokens, device):
            return self.embeds, None, None, self.info

    def vision(index, grid_h, grid_w):
        size = (grid_h // 2) * (grid_w // 2)
        return {"type": "image", "index": index, "size": size,
                "extra": {"grid": torch.tensor([[1, grid_h, grid_w]]),
                          "deepstack": [torch.randn(size, hidden) for _ in range(3)]}}

    def run(span_lengths, blocks, prompt, shift=0):
        """`blocks[i]` is span i's vision block as (index in the span, grid h,
        grid w), a list of them for a span that holds several, or None.
        Returns (largest difference over largest value per part, tags
        equal)."""
        starts = [sum(span_lengths[:i]) for i in range(len(span_lengths))]
        total = sum(span_lengths) + prompt
        embeds = torch.randn(1, total, hidden)
        local = [[vision(*one) for one in (b if isinstance(b, list) else [b])] if b else [] for b in blocks]
        whole = [dict(v, index=v["index"] + st) for per_span, st in zip(local, starts) for v in per_span]
        with torch.inference_mode():
            if whole:
                position_ids, visual, deepstack = Qwen3VL.build_image_inputs(None, embeds, whole)
            else:
                position_ids = visual = deepstack = None
            core = lm(None, embeds=embeds.clone(), position_ids=position_ids, embeds_info=whole,
                      visual_pos_masks=visual, deepstack_embeds=deepstack, dtype=torch.float32)[0]
            spans, images = [], []
            for st, n, v in zip(starts, span_lengths, local):
                span = enc._encode_span(Embedded(embeds[:, st:st + n].clone(), v),
                                        lm, [(0, 1.0)], st, images, list(spans), device)
                spans.append(span)
                images = images + [dict(e, index=e["index"] + st) for e in span.images]
            encoded = enc.EncodedReferences(spans=spans, ref_blocks=[], ref_items=[], clip=None, patches=None)
            start = encoded.length
            positions = enc._positions(encoded.images(), start, prompt + shift, device)[:, shift:]
            x, _ = enc._lm_pass(lm, embeds[:, start:].clone(), positions,
                                past=lambda i: enc._kept_layer(spans, i, device), past_len=start, keep=False)
        split = torch.cat([sp.rows for sp in spans] + [x], dim=1)
        cuts = starts + [start, total]
        ratios = [float((core[:, a:b] - split[:, a:b]).abs().max() / core[:, a:b].abs().max())
                  for a, b in zip(cuts, cuts[1:])]
        from comfy.text_encoders.minimax import token_tags_from_embeds_info
        tags = torch.equal(token_tags_from_embeds_info(total, whole),
                           token_tags_from_embeds_info(total, encoded.images()))
        return ratios, tags

    for name, args in (
            ("two vision spans and a prompt", ([11, 11], [(4, 4, 6), (4, 6, 4)], 9)),
            ("a label-only span ahead of a vision span", ([5, 13], [None, (4, 4, 8)], 7)),
            ("a one-token prompt", ([5, 13], [None, (4, 4, 8)], 1)),
            # the second span is a video reference's shape: label and timestamp text, then
            # three vision blocks with text between them
            ("a still's span, then a span of three vision blocks with text between, and a prompt",
             ([11, 26], [(4, 4, 6), [(3, 4, 6), (11, 4, 6), (19, 4, 6)]], 9))):
        ratios, tags = run(*args)
        check(f"{name}: every span and the prompt within the bound, tags equal",
              max(ratios) < SMALL_MODEL_TOL and tags, f"largest {max(ratios):.1e}")
    ratios, _tags = run([11, 11], [(4, 4, 6), (4, 6, 4)], 9, shift=1)
    check("RED CONTROL: prompt positions off by one are far outside the bound",
          ratios[-1] > 100 * SMALL_MODEL_TOL, f"prompt rows {ratios[-1]:.1e}")


def load_pack():
    from _lib import bootstrap
    # The ComfyUI root, which this check used to need from PYTHONPATH, and
    # core's CPU path when no card is visible (unchanged: the same rule).
    bootstrap(front=False)
    spec = importlib.util.spec_from_file_location(
        "h3x", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["h3x"] = module
    spec.loader.exec_module(module)
    return importlib.import_module("h3x.reference_encode")


def main() -> int:
    import torch
    enc = load_pack()
    failures = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("the copied loop's source:")
    import comfy.text_encoders.llama as llama
    for name, pinned in PINNED.items():
        cls, fn = name.split(".")
        got = hashlib.sha256(inspect.getsource(getattr(getattr(llama, cls), fn)).encode()).hexdigest()[:16]
        check(f"{name} is the source the copy was made from", got == pinned,
              "" if got == pinned else f"now {got}, pinned {pinned}: re-read it, re-run "
              "bench/measure_reference_split.py, then move the pin")

    print("\nthe loop against core on a small random model:")
    small_model_cases(enc, check)

    print("\npositions:")
    from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids
    grid_a, grid_b = torch.tensor([[1, 8, 12]]), torch.tensor([[1, 6, 4]])
    images = [{"type": "image", "index": 5, "size": 24, "extra": {"grid": grid_a}},
              {"type": "image", "index": 37, "size": 6, "extra": {"grid": grid_b}}]
    total, cut_a, cut_b = 60, 30, 44                   # span one, span two, then the prompt
    whole = qwen2vl_mrope_position_ids(images, total, "cpu")
    first = enc._positions(images[:1], 0, cut_a, "cpu")
    second = enc._positions(images, cut_a, cut_b - cut_a, "cpu")
    prompt = enc._positions(images, cut_b, total - cut_b, "cpu")
    check("three cuts of the sequence carry core's positions",
          torch.equal(torch.cat([first, second, prompt], dim=1), whole),
          f"{tuple(whole.shape)}")
    plain = enc._positions([], 7, 5, "cpu")
    check("no vision block ahead: a plain count from the start offset",
          torch.equal(plain, torch.arange(7, 12).unsqueeze(0)))

    print("\nthe causal mask:")
    for seq, past in ((7, 0), (5, 11), (1, 9)):
        total = past + seq
        square = torch.empty(total, total).fill_(torch.finfo(torch.float32).min / 4).triu_(1)
        check(f"{seq} new tokens on {past} kept: the rows of core's square mask",
              torch.equal(enc._causal_rows(seq, past, torch.float32, "cpu"), square[past:]))

    print("\nthe span cut:")
    kinds = lambda *k: [{"type": t} for t in k]        # noqa: E731
    check("a still, a sounded video, a still: three spans, the soundtrack's label with its video",
          enc.span_groups(kinds("image", "audio", "video", "image")) == [[0], [1, 2], [3]])
    check("trailing audio references form a last span of their own",
          enc.span_groups(kinds("image", "audio", "audio")) == [[0], [1, 2]])

    print("\nnothing is kept between runs:")
    rc = importlib.import_module("h3x.reference_conditioning")
    check("the module holds no store and no key function",
          not any(hasattr(enc, name) for name in ("STORE", "_Store", "span_key", "STORE_BYTES")))
    check("the one-node path has no keep_references",
          all(i.id != "keep_references" for i in rc.MiniMaxH3ReferenceConditioning.define_schema().inputs))

    print("\nthe report's labels:")
    rc_ = importlib.import_module("h3x.reference_conditioning")
    report = importlib.import_module("h3x.reference_report")
    sound = {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000}
    records = (rc_.RuntimeVideoReference(frames=torch.rand(24, 64, 96, 3), loaded_fps=24.0, soundtrack=sound),
               rc_.RuntimeImageReference(image=torch.rand(1, 64, 96, 3), size_policy="match"))
    priced = report.price_references(records, 96, 64, 5)
    got = [getattr(item, "label", None) for item in getattr(priced, "items", [])]
    check("a still after a sounded video is <Picture 1>, not the video's label",
          got == ["<Video 1>", "<Picture 1>"], f"{got}")

    print("\nthe two nodes:")
    nodes = importlib.import_module("h3x.nodes")
    source = inspect.getsource(nodes)
    check("both are in the pack's node list",
          # in the list itself, not the import line: 2026-10-04, a node appended after them
          re.search(r"^\s+MiniMaxH3EncodeReferences, MiniMaxH3PromptOnReferences[,\]]", source, re.M) is not None)
    a, b = enc.MiniMaxH3EncodeReferences.define_schema(), enc.MiniMaxH3PromptOnReferences.define_schema()
    check("both are marked experimental", bool(a.is_experimental and b.is_experimental))
    out_type = a.outputs[0].io_type if hasattr(a.outputs[0], "io_type") else getattr(a.outputs[0], "get_io_type", lambda: None)()
    in_types = [getattr(i, "io_type", None) or getattr(i, "get_io_type", lambda: None)() for i in b.inputs]
    check("the prompt node takes the encode node's output type",
          out_type is not None and out_type in in_types, f"{out_type}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {', '.join(failures)}")
        return 1
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
