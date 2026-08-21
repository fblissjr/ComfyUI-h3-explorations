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

**But the rows are untrained, so routing to them reaches init noise.** Measured
2026-08-21 by `bench/audit_h3_token_embeddings.py`, record at
`bench/results/2026-08-21_h3_token_embeddings.json`, on the official release
and on both repacked encoders. Read against two controls -- the stock Qwen
special tokens, which are certainly trained, and the padding rows past 151676,
which certainly are not -- the seven land on the untrained pole in all three
files, at roughly a hundredth of the distance to the trained one. The
controls separate cleanly, so the measurement discriminates.

This is the first thing to know before treating the tokenizer gap as a fidelity
defect, and it cuts in an unobvious direction. It does **not** make ComfyUI's
behaviour equivalent to the release: the release emits one ID and attends an
untrained vector, ComfyUI emits several trained BPE pieces, and those are
different sequences of different lengths, which shifts every position after
them. What it does mean is that **making the markers reachable is a parity fix,
not a quality fix** -- neither path carries learned meaning for `<d>` itself,
so nobody should expect the marker to start working once it routes. Two
readings remain open and this measurement does not separate them: the tokens
may be vestigial, or the text encoder may simply have been frozen through H3
training, in which case no row moved and the norm says nothing about whether
MiniMax intended them. `vendor_tokens.py` is the node that makes them
reachable, and it is correct to keep the caveat.

**`<d>` is not exotic, and a shipped graph already uses it.** Corrected
2026-08-21; this section previously said no prompt in this repo used any of the
seven, which was written from expectation rather than a grep. The official
prompt guide requires `<d>[Language] ...</d>` for **all** dialogue and lyrics
(`coderef/MiniMax-H3/skills/h3-prompt-writing/references/ref-en.txt:220`),
preserves source language only inside it (`:5`), forbids repeating its contents
in the audio sections (`:303`), and uses it throughout its own worked examples
(`:330-332`). `workflows/build_workflows.py:1718` emits it into
`workflows/h3_ref_audio_voice.json` and its API twin, quoting that rule in the
comment above. So every H3 prompt containing speech is affected on ComfyUI —
a prompt following the vendor's guide is precisely the prompt that trips this.

**Measured 2026-08-21** by `bench/compare_h3_tokenizers.py`, whose record is
`bench/results/2026-08-21_h3_tokenizer_markers.json`. Both tokenizers loaded
through `AutoTokenizer`, the
release from `coderef/MiniMax-H3/tokenizer/` (byte-identical to the copy beside
the weights) and ComfyUI from its bundled directory, `add_special_tokens=False`:

| string | release | ComfyUI |
|---|---|---|
| `<d>` | `[151669]` | `[90707, 29]` |
| `</d>` | `[151670]` | `[522, 67, 29]` |

On the dialogue line the voice graph actually ships, the release emits 14 IDs
opening `151669` and closing `151670`; ComfyUI emits 15, and its split does not
respect the marker boundary — the language tag's `[` is fused into the opening
debris and the sentence's final `.` into the closing debris. `<Picture 1>: `,
`<Video 1>: `, `<Audio 1>: `, `<0.2 seconds>` and ordinary prose are
ID-identical on both sides, which is why this hides.

Worth noticing: `bench/check_prompt_guide_conformance.py` and
`bench/preflight_graph.py` both grade `<d>` placement and grammar. They are
enforcing the syntax of a marker the tokenizer downstream cannot express.

**Still not known:** what the other six markers are *for*. The lyrics and
caption pairs suggest structured audio-and-caption conditioning, and no prompt
here reaches for them.

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
