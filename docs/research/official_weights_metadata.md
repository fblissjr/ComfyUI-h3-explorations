# What the official release ships, against what ComfyUI does with it

last updated: 2026-08-25

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

### 1. Seven H3-specific special tokens were unreachable in ComfyUI

**Fixed natively in ComfyUI by merged PR 15808.** This installed checkout
contains the fix as commit `924743af`; it declares the seven on a
`Qwen3VLSDTokenizer` subclass so every consumer gets them -- core's
`MiniMaxH3ReferenceToVideo` included. This is a native ComfyUI resolution, not
a fix supplied by this custom-node repo. An older install without that commit
still has the defect and this section describes what that install does.
`bench/audit_h3_marker_tokenization.py` is this repo's verification harness. It
now requires the native tokens and reconstructs the legacy tokenizer only as a
measurement control.

The local fallback is retired. Both custom conditioners rely on native
ComfyUI; their `vendor_tokens` schema slots and the standalone
`MiniMaxH3VendorTokens` node were kept inert for saved-graph loadability and
were **removed on 2026-08-27** by owner decision.

The finding below stands as written for an unpatched install.

The release's `tokenizer/tokenizer_config.json` declares twenty
`additional_special_tokens`. ComfyUI's bundled `comfy/text_encoders/qwen25_tokenizer/tokenizer_config.json`
declares thirteen. ComfyUI's H3 tokenizer resolves to that bundled directory
(`comfy/text_encoders/qwen3vl.py:149`), so this is the file in play for every
H3 prompt. **The seven the release adds on top of stock Qwen3-VL: `<d>`,
`</d>`, `<|cutoff|>`, `<|lyrics_start|>`, `<|lyrics_end|>`,
`<|caption_start|>`, `<|caption_end|>`.** They were absent from ComfyUI when
this section was written and are native now, owned by `MiniMaxH3Tokenizer`; the
paragraphs below describe the gap and are kept for why it existed, not as
current state.

This is what the model card means when it says the tokenizer and configuration
files provided in the H3 repository are required.

**The damage is not confined to the marker** — historical, measured
2026-08-22, before the tokens were native. BPE has no
reason to stop at the angle bracket, so the fragments fuse with the text on
either side: the release emits `<d>` then `[`, ComfyUI emits `>[` as one token,
and a sentence-final `.` is dragged forward into `.</`. Ordinary prose tokens
next to a marker come out different too. `bench/results/2026-08-24_h3_marker_tokenization_native.json`
carries the per-scene counts, and the reference presentation is covered there as
well -- the marker reaches the encoder beside the vision blocks, so reference
prompts carrying dialogue are exposed on the same terms as text ones.

### The directory name is a misnomer and the vocabulary is right

**Read `qwen25_tokenizer` and the obvious conclusion is that ComfyUI points a
Qwen2.5 tokenizer at a Qwen3-VL model. That is not what is happening**, and the
distinction decides how large the defect is. Measured 2026-08-21 against a
stock Qwen3-VL checkout, which is the arm that settles it:

| | stock Qwen3-VL | ComfyUI bundled | H3 release |
|---|---|---|---|
| `tokenizer_class` | `Qwen2Tokenizer` | `Qwen2Tokenizer` | `Qwen2Tokenizer` |
| `vocab.json` entries | 151,643 | 151,643 | 151,643 |
| vocab identical to bundled | **yes** | — | yes |
| `merges.txt` identical to bundled | **yes** | — | yes, discounting a `#version` header |
| `added_tokens_decoder` | 26 | 26 | 26, agreeing on content and id |
| `additional_special_tokens` | 13 | 13 | **20** |

ComfyUI's bundled directory is byte-equivalent to **stock Qwen3-VL's**
tokenizer, not to some older Qwen2.5 one. Qwen2.5, Qwen3 and Qwen3-VL ship one
BPE vocabulary between them, and the H3 release's own config names
`Qwen2Tokenizer` as its class. So the name is legacy and the vocabulary is
correct. **The entire divergence is the last row**: the release adds seven
entries on top of stock Qwen3-VL and ComfyUI has no way to see them. Ordinary
prose tokenizes identically on both sides; only the seven markers do not.

### Why ComfyUI cannot load them, which is structural rather than careless

`Qwen3VLSDTokenizer` hardcodes the bundled path with no override
(`comfy/text_encoders/qwen3vl.py:149`). More to the point, there is nowhere
else to look: ComfyUI loads a single-file `.safetensors` text encoder, not an
HF model repo, so no `tokenizer_config.json` travels with the weights and
nothing can carry a model's own additions. Bundling one directory works for the
entire Qwen family precisely *because* the vocabulary is shared, and it works
for every case except a model that adds tokens of its own. H3 is that case, and
the model card's sentence is aimed at exactly it. The native
`MiniMaxH3Tokenizer` subclass is now the answer; the retired local node does
not alter tokenizers.

**What happens instead — REMOVED 2026-08-26, it disagreed with the rest of
this section and with the code.** It said ComfyUI never appends the seven, so
`<d>` tokenized as ordinary BPE pieces. `comfy/text_encoders/minimax.py:32`
defines `MINIMAX_EXTRA_TOKENS` with all seven at the release's own ids, and
`MiniMaxH3Tokenizer` is the class in play. Measured the same day through that
tokenizer, which is the observable the rule says to check rather than a branch
name or a commit:

```
"<d>[English] Then I will take two.</d>"
  -> [151669, 58, 22574, 60, 5005, 358, 686, 1896, 1378, 13, 151670]
      <d>     [  English ]  Then  I   will  take  two  .   </d>
```

One id each, and no fusion with the neighbouring text. **Dialogue markers work;
write them.**

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
them. What it does mean is narrower than it first looks, **and narrower than this
paragraph claimed when it was written earlier the same day.** The first version
said making the markers reachable is "a parity fix, not a quality fix, so
nobody should expect the marker to start working once it routes." That
conflates two different places a token can carry meaning. The measurement
constrains the *encoder*: Qwen assigns row 151669 no learned semantics. It says
nothing about the *DiT*, which was trained on Qwen hidden states produced from
prompts tokenized by the release tokenizer -- and the prompt guide mandates
`<d>` for all dialogue, so those training prompts almost certainly contained
it. An untrained embedding is still a fixed, distinctive vector, and a frozen
encoder turns it into a fixed, distinctive hidden state. That is exactly what a
downstream model can learn to read as a delimiter.

So the honest position is: **the marker is meaningless to Qwen and may well be
load-bearing to the DiT, and the embedding audit cannot tell you which.**

**What the encoder does with it, measured 2026-08-21** by
`bench/grade_h3_marker_tokens.py`, record at
`bench/results/2026-08-21_h3_marker_token_states.json`. Three arms per prompt on
the same weights -- markers routed to their real ids, markers as the BPE
fragments ComfyUI emits, and markers deleted -- compared over the token spans
whose ids agree, because the arms have different lengths and a positional diff
would compare unrelated tokens.

| prompt | vendor vs comfy | vendor vs stripped | ratio |
|---|---|---|---|
| one pair, short | 0.102 | 0.159 | 0.64 |
| the shipped voice line | 0.034 | 0.037 | 0.90 |
| fourteen pairs, t2v format | 0.091 | 0.107 | 0.85 |

Relative L2 on the layer-50 states. **The deleted arm is what makes these
readable**: any change to a token sequence moves the states somewhat, so
vendor-against-comfy alone says nothing. Read against it, ComfyUI's fragments
recover between a tenth and a third of what the marker does, and on the two
strongest arms about a tenth. **The cheap escape hatch is closed** -- "the
fragments carry the delimiter anyway, so routing them is cosmetic" is refuted.

Two controls, both held: a marker-free prompt tokenizes to identical ids and
identical states on both sides, and the forward is bit-deterministic. Without
the first, the harness could be measuring something other than the markers.

**This still measures the encoder, not the DiT.** It establishes that the
representation genuinely changes and that comfy's version is much nearer to
"no marker" than to the release's. Whether the DiT reads the difference needs a
comparison at its boundary -- not a rendered pair, which `CLAUDE.md`'s
different-sample rule rules out. Two
readings remain open and this measurement does not separate them: the tokens
may be vestigial, or the text encoder may simply have been frozen through H3
training, in which case no row moved and the norm says nothing about whether
MiniMax intended them. Native ComfyUI makes them reachable; it is correct to
keep the caveat.

**And the reason is the encoder is stock, closed 2026-08-21.** The release's
README says the H3-Encoder "uses the full pretrained weights of Qwen3-VL-32B"
and taps layer 50 (`coderef/MiniMax-H3/README.md:128`). Run against a stock
Qwen3-VL checkout H3 never touched, rows 151669-151675 sit on that model's own
padding pole too. So the rows are untrained because **nobody ever trained
them**: they are Qwen's padding, and MiniMax added seven tokenizer entries
pointing at them without training the encoder. That settles the two readings
this section listed as open -- it is the frozen-encoder one, and the markers
carry no learned meaning in any implementation, the vendor's included.

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

### How the fixed install realises them, and why the vision side cannot shift

Read from the installed checkout and exercised once on 2026-08-25, because
the question "don't the tokenizer, the embedding and the vision tower all have
to agree on `<d>`'s id" has a precise answer and the intuitive one is wrong.

**The ids are assigned by append order, not by the constants.**
`MiniMaxQwenSDTokenizer` (`comfy/text_encoders/minimax.py`) takes the stock
bundled tokenizer and calls `add_special_tokens({"additional_special_tokens":
[the seven]})`; HF gives appended special tokens the next free ids in list
order. Observed: the stock tokenizer has length 151,669 with `</think>` at
151,668; after the call it has 151,676, with `<d>` at 151669 through
`<|caption_end|>` at 151675. The `MINIMAX_EXTRA_TOKENS` dict in that file
records this outcome; nothing assigns from it. The release lands on the same
ids because it performed the same append. Every existing id, the four vision
sentinels included, keeps its number, which is what "the ids do not shift"
means. A prompt through it: `<d>[English] Hello there.</d><|cutoff|>` becomes
`[151669, 58, 22574, 60, 21927, 1052, 13, 151670, 151671]`, one id per marker
and the full stop kept as its own token.

**The picture is never a token.** `MiniMaxH3Tokenizer.tokenize_with_weights`
builds a flat list: text segments as ids, and each image or two-frame video
block as three entries, the int `151652`, a dict holding the pixels, the int
`151653`. Labels such as `<Picture 1>: ` and `<0.5 seconds>` are ordinary BPE
text; the prompt comes last.

**Ids become rows; pictures become vectors; they are spliced by position.**
`comfy/sd1_clip.py::process_tokens` looks the integer entries up in the
`[151936, 5120]` table (`<d>` reads row 151669, the sentinels rows 151652 and
151653) and hands each dict to `preprocess_embed`, which runs the image
processor and the vision tower and returns merged `[N, 5120]` features plus
DeepStack features. Those features are concatenated into the embedding
sequence at the dict's list position, between the two sentinel rows, and
`embeds_info` records the span. **Native ComfyUI's H3 path never materialises
`<|image_pad|>` (151655) at all**: there are no pad ids to scatter into. The
Transformers path the calibration lane drives does the opposite, expanding real
pad ids and scattering features into them; Gate 1
(`docs/research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_gate1_seam_acceptance.md`)
proved the two produce identical token streams and vision spans. The repo's
AWQ loader keeps the native splice and replaces only `preprocess_embed`.

**The model consumes positions, not ids.** `Qwen3VL.build_image_inputs`
derives M-RoPE position ids and a boolean `visual_pos_masks` from
`embeds_info`; after decoder layers 0, 1 and 2, `comfy/text_encoders/llama.py`
adds the DeepStack features at `x[visual_pos_masks]`. The vision tower has no
vocabulary to align. It needs its output spliced at the right positions, and
the positions come from the list.

**So what has to agree, and what asserts it.** The two hardcoded sentinel ids
must be the rows the release trained as vision start and end, and the seven
appended ids must land on the seven rows the release names. Both hold by
construction, because the seven were appended above everything that existed,
and both are checked: `bench/audit_h3_marker_tokenization.py`'s marker-free
control shows the vision structure byte-identical with and without the fix
over two images, an odd-frame video and an audio reference, and
`h3_awq_encoder.py::_validate_native_tokenizer` refuses a load whose ids or
config sentinel roles disagree. **Nothing here is broken.** What id agreement
cannot settle is upstream of ComfyUI: whether MiniMax trained the DiT against
these dedicated ids or against legacy BPE fragments, which the canonical
baseline keeps as UNKNOWN, and the fact that the seven rows themselves are
untrained, measured above.

### 2. The native processor's pixel bounds are Qwen2-VL's defaults

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

## AWQ checkpoint formats: one native, one locally adapted

Recorded 2026-08-23 after a filename-level comparison incorrectly suggested
that ComfyUI either supports “AWQ” or does not. AWQ describes calibration; the
two installed artifacts use different storage and loader contracts.

| artifact | stored contract | owner here |
|---|---|---|
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | native H3 namespace; 350 `format=nvfp4` `comfy_quant` linears plus an int8 embedding | core `CLIPLoader` — native ComfyUI support |
| `qwen3vl_32b_minimax_h3_w4a16_awq.safetensors` | full 64-layer HF namespace; compressed-tensors signed W4 group-128 `weight_packed`/`weight_scale`/`weight_shape` records | this repo's `MiniMaxH3AWQEncoderLoader` |

Core's directory scan offers both names, but discovery is not representation
support. Executing the W4A16 artifact under core on 2026-08-23 selected
Qwen3-VL-8B from its full HF namespace, instantiated width 4096, and rejected
the H3 width-5120 tensors. The local adapter is not bound to that basename: it
validates whichever file the user selects, including its embedded config and
complete native H3 tensor inventory, against
[`config/qwen3vl_32b_minimax_h3_w4a16_awq/`](../../config/qwen3vl_32b_minimax_h3_w4a16_awq/),
retains layers 0–49, maps the namespace, exposes the packed nibbles to
comfy-kitchen without a weight-sized copy, and uses the source processor
configuration. It deliberately leaves core architecture and tokenizer in
charge. [`bench/check_h3_awq_encoder.py`](../../bench/check_h3_awq_encoder.py)
proves the native NVFP4 control and the local W4A16 contract separately.

---

## Which divergence is live in which mode

Added 2026-08-21. Everything above and in [`h3_references.md`](../h3_references.md)
is stated as a property of the pipeline. This section answers the question that
actually decides what to do about any of it: **for the mode you are running,
which of these fire at all?** Several do not, and the inert ones are the useful
part -- they are the ones both outside reviews of this pipeline flagged as
concerns on paths where they cannot bite.

| divergence | t2v | first frame | last only | first+last | ref2v |
|---|---|---|---|---|---|
| the seven markers | dialogue prompts only | same | same | same | same |
| processor pixel bounds | — | **inert** | **inert** | **inert** | live for small/extreme stills and the bounded duration-aware video regime |
| bilinear against the release's processor resize | — | geometry-inert | geometry-inert | geometry-inert | live whenever the differing policy resizes; independent layer-49 contribution unmeasured |
| VAE posterior mean against seed-42 sample | — | live | live | live, twice | live, every reference |
| crop against stretch policy | — | — | **live** | — | — |
| stretch to a fixed canvas | — | live without `KeyframeCanvas` | live, **not closable** | live without `KeyframeCanvas` | — |
| reference sizing, fps, audio duration, mono | — | — | — | — | live |
| no partition admission | live | live | live | live | live |

**t2v is the cleanest path in the model.** The vision tower never executes, so
nothing about bounds, interpolation, posteriors, crops or references applies.
The only thing that reaches it is the marker gap, and only when the prompt
carries dialogue -- which the guide requires be written as `<d>`. On a
descriptive prompt the two implementations agree.

**The processor bounds are inert on every keyframe mode, and that is measured
rather than assumed.** A keyframe arrives on a legal H3 canvas; every legal
canvas is a multiple of 32, which is exactly the `patch_size * merge_size`
factor the Qwen helper rounds to; and every legal canvas sits between both
implementations' floors and both ceilings. So the helper's resize is a no-op,
no interpolation happens, and the bilinear-against-bicubic difference never
fires. It is a real divergence on a path the keyframe modes do not take.

**What first-frame actually pays is the posterior.** ComfyUI takes the mean,
the release samples at a pinned seed. The condition rows carry the shipped
0.999 aug, so they reach the DiT essentially clean and the difference is not
washed out by noise -- it is what your first frame *is*. Read it as adherence
at frame 0, not as an artifact.

**Last-only is the most divergent keyframe mode and the only one nothing here
closes.** ComfyUI picks crop-against-stretch from which *socket* was wired, so
`last_frame` takes a cover crop into a canvas fixed elsewhere, and loses
whatever falls outside that aspect. `MiniMaxH3KeyframeCanvas` requires a first
frame and cannot reach this case, so rendering last-only from a non-16:9 source
silently crops content.

**The vendor side, read 2026-08-21 and not what this paragraph first said.** It
originally claimed sglang anchors on the first *presented* keyframe, taken from
the Codex review rather than from source. `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/prequeue.py:97-107` does
something different and its own comment says so: "Select by semantic time, not
request/material iteration order." Geometry is *deferred* and the canvas is
resolved from the anchor's own display shape, so with one keyframe that
keyframe **is** the anchor whichever socket it came from, and the canvas is
derived from it. The consequence is stronger than the original claim: the
release does not stretch or crop a last-only keyframe, because the canvas is
built to fit it. ComfyUI starts from a canvas and makes the image fit instead.

**first+last is the closest mode to release behaviour** -- first stretched,
last cropped, matching the vendor structure. It pays the posterior twice and
little else.

**ref2v collects everything**, and pays token-level differences roughly twice
over because a reference image costs in the text segment as well as the
reference segment. It is the mode where both still-image boundary behavior and
the separate clip-wide video-processor policy can be live;
[`h3_references.md`](../h3_references.md) owns the measured thresholds. This
repo's shipped typed graphs use `video_policy=encoder` (all but the `release`
probe arm), so the native stock video exposure remains a vendor gap rather
than their runtime path.

---

## Not done

The end-to-end conditioning trace this file was opened for is further along
than it was. What is still unread: both VAE decodes followed to the end, and
what differs between the `int8_convrot` and `nvfp4_awq` text encoders beyond
the quantisation itself -- their embedding tables were compared for the seven
marker rows on 2026-08-21 and agree, which is one tensor of many.
