# What the official release ships, against what ComfyUI does with it

last updated: 2026-08-21

The MiniMax H3 release as published, read from the repo on the `Storage` side
on **2026-08-21**: `model_index.json`, both partition entry points, and every
`config.json`, `preprocessor_config.json` and tokenizer file beside the
weights, compared against what ComfyUI hard-codes. Source reads only; nothing
here was rendered.

**Scope.** This file owns the *metadata* comparison — what is declared in the
release against what the code assumes. The runtime and optimization comparison
against sglang is [`sglang_comparison.md`](sglang_comparison.md); the
reference-conditioning path is [`h3_references.md`](../h3_references.md).

---

## Findings, worst first

### 1. Seven H3-specific special tokens are unreachable in ComfyUI

The release's `tokenizer/tokenizer_config.json` declares twenty
`additional_special_tokens`. ComfyUI's bundled `comfy/text_encoders/qwen25_tokenizer/tokenizer_config.json`
declares thirteen — the stock Qwen2.5-VL set. ComfyUI's H3 tokenizer resolves
to that bundled directory (`comfy/text_encoders/qwen3vl.py:149`), so this is
the file in play for every H3 prompt. **Present in the release and
absent from ComfyUI: `<d>`, `</d>`, `<|cutoff|>`, `<|lyrics_start|>`,
`<|lyrics_end|>`, `<|caption_start|>`, `<|caption_end|>`.**

This is what the model card means when it says the tokenizer and configuration
files provided in the H3 repository are required.

Everything *else* about the two tokenizers agrees, which is why this is easy to
miss and why it is the only thing that matters here. Verified by comparison:
`vocab.json` is dict-identical at 151,643 entries; `merges.txt` is identical
line for line once ComfyUI's extra `#version` header is discounted; and the 26
entries of `added_tokens_decoder` agree on both content and id. So ordinary
prose tokenizes identically on both sides. Only the seven markers do not.

**What happens instead.** None of the seven is in the base vocab or in
`added_tokens_decoder`, so the release's loader appends them past the end of
the vocabulary. ComfyUI never appends them, so a prompt containing `<d>` is
tokenized as ordinary text — the angle brackets and the letter, as several BPE
pieces — rather than as one marker.

**The weights can hold them.** The release's `text_encoder/config.json`
declares
`vocab_size: 151936`, and both repacked encoders on this box carry
`model.embed_tokens.weight` at `[151936, 5120]` — the full table, with room
past the 151,669 the vocabulary and added tokens occupy. So the embedding rows
for these markers ship with the files we already run. What is missing is the
tokenizer entry that would route to them.

**Not yet known, and worth knowing before anyone acts on this:** what the seven
markers are *for*. The lyrics and caption pairs suggest structured
audio-and-caption conditioning; `<d>` is undocumented beyond being listed. No
prompt in this repo uses any of them, so nothing measured here is affected —
but any prompt-format work that reaches for them is currently writing text.

### 2. The tokenizer's pixel bounds are Qwen2-VL's defaults

Owned by [`h3_references.md`](../h3_references.md), which carries the table and
the consequence. Recorded here only because it is the same class of defect
found the same way: the geometry is read correctly and two constants next to it
are inherited from a shared helper's signature.

### 3. The partition split is in the weights

The release ships shared components plus two partition entry points, each with
its own `model_index.json` carrying a `_minimax_h3` block. The fl2va one
declares `tasks: ["t2va", "fl2va"]`; the ref2va one declares `tasks:
["ref2va"]`. That is the vendor stating ref2va serves no text-to-video task —
sglang's refusal reads that field rather than inventing the rule. Both blocks
also carry `sigma_shift_scales` of 12.0 video and 3.0 audio, matching the two
scheduler configs and what this repo ships.

---

## Checked and clean

Recorded so nobody re-derives them:

- **`transformer/config.json` and `transformer_ref/config.json` are
  identical.** fl2va and ref2va differ in weights alone, not in declared
  architecture.
- **The layer-50 tap matches.** The release's text encoder declares 64 hidden
  layers; ComfyUI's `Qwen3VL_32BConfig`
  (`comfy/text_encoders/llama.py:288-296`) truncates to the first 50 with no
  final norm and no `lm_head`, which is the vendor's own description of the H3
  encoder. We are not paying for fourteen unused layers.
- **Patch geometry matches**, including the `patch_size=16` that ComfyUI passes
  explicitly (`comfy/text_encoders/qwen3vl.py:62-68`) rather than inheriting
  the shared helper's 14, and the 0.5 mean/std normalisation that separates
  Qwen3-VL from Qwen2.5-VL's CLIP statistics.
- **Scheduler shifts match** the release: 12.0 video, 3.0 audio.

---

## Not done

The end-to-end conditioning trace this file was opened for is not finished.
What exists is the metadata layer. Still unread: the four conditioning paths
(pure t2v, first frame, keyframes, references) followed from model load to both
VAE decodes, and what differs between the `int8_convrot` and `nvfp4_awq` text
encoders beyond the quantisation itself.
