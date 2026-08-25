# A persistent conditioning cache: design note

Written 2026-08-25, in the reference-nodes lane, at the v2 lead's request.
**Design only; nothing here is built.** It argues for one thing, the
cross-session case, and says where ComfyUI already covers the rest.

## What it would save, and what it would not

The encode itself is cheap: the layer-49 prefill on a reference-bearing
prompt is seconds on this card. What costs is getting the encoder resident:
19 GB as W4A16, competing with the DiT for 24 GB, so ComfyUI evicts and
reloads it around every encode. MEASURED, the conditioning node (node 5 in
every generated graph; it also performs the reference VAE encodes) in the
2026-08-18 to 2026-08-22 arm rows under `bench/results/`: 60 s of a 1,607 s
all-references render, 21 s of a 281 s multi-reference render, 2.5 s of a
160 s turbo render. So the prize per cold encode is tens of seconds, and a
render whose conditioning already exists never needs the encoder resident at
all, which is also the memory headroom argument.

## Where ComfyUI's own cache already covers it

`comfy_execution/caching.py`: the default cache (`HierarchicalCache`) keys a
node's output on the signature of its inputs and every ancestor's, so within
one server session a prompt, media and policy set that did not change is not
re-encoded. Changing the seed, sampler, steps or LoRA touches nothing upstream
of node 5 and hits. SOURCE. Two cases fall outside it:

- **Across sessions.** Nothing persists; a restart re-encodes everything.
- **Bench mode.** `start.sh nodynvram` (alias `bench`) runs `--cache-none`,
  so every `run_graph_arms.py` render re-executes node 5, which is why the arm
  rows above carry a node-5 cost on every seed of every arm. A matched-seed
  batch of K seeds by N arms pays K x N encodes today where it needs N.

Those two are the whole case. A same-session, default-mode workflow gains
nothing from this.

## The key, which is the whole design

An entry is valid only for exactly the inputs that produced it. The key is
every input the encode depends on, and **a key with any field missing refuses
rather than matches**; that rule is the difference between a cache and a
source of silently wrong conditioning.

| field | why it is in the key | where it already exists |
|---|---|---|
| encoder artifact digest | v1, v2 and any two calibrations of the same architecture produce different states from the same input; a stamped contract alone does not separate two calibrations that declare the same bounds | the loader's full-file digest control |
| encoder contract | `reference_geometry.ENCODER_CONTRACT_KEYS`: `source`, image and video bounds and geometry; decides the stage-two view under `encoder` policy | stamped on the CLIP by `MiniMaxH3AWQEncoderLoader`; read by `encoder_contract_from_clip` |
| prompt bytes (sha256) | the text is the presentation | capture manifest |
| ordered media, each by sha256, role and label | order is part of the presentation (`<Picture i>` numbering), and the same file as keyframe and as reference sizes differently | capture manifest `references[].sha256` |
| per-reference sizing knobs | `size_policy`, `allow_upscale`, `short_edge`, `qwen_short_edge`: the same file at two views is two encodes | the append node's inputs |
| `video_policy`, `image_policy` | select whose bounds apply at stage two | the conditioner's inputs |
| canvas and length | reference videos are sized by the canvas rule and frame count; the emitted AV latent's shape follows them | the conditioner's inputs |
| audio references (label only) | a text label enters the presentation | the append node |
| tokenizer identity | the seven H3 special tokens are appended by the installed ComfyUI, and their ids are what the presentation is made of | `vendor_tokens.py` / the loader's token-set assertion |

Every row is something this repo already computes or records for the capture
manifest (`docs/capture_manifest_schema.md`), so the key is the manifest's
provenance record reused, not a new schema.

## The payload

What node 5 emits, and nothing else:

- the conditioning list: the layer-49 hidden state, `[1, L, 5120]` in BF16
  (about 100 MB at 10k tokens), with its dict, which carries
  `minimax_token_tags` from the encoder and `minimax_refs` from the
  conditioner (the reference latents the DiT re-injects every step, so they
  must travel with the state, not be recomputed from a possibly different
  VAE);
- the AV latent, which is a shape and is recomputed from canvas and length
  rather than stored.

Stored as one safetensors file per entry with the key fields as its JSON
metadata, under a cache directory that is not `models/` and not the media
share.

## The node shape

One wrapper node, not a Save/Load pair. `MiniMaxH3CachedReferenceConditioning`
takes exactly the inputs of `MiniMaxH3ReferenceConditioning` plus a cache
directory, computes the key from those live inputs, and on a hit returns the
stored conditioning without touching the encoder; on a miss it runs the
existing node and stores the result. Reasons over a Save/Load pair: the key is
never typed by a person, so it cannot name the wrong entry; a graph is one node
swap; and the miss path *is* the existing node, so there is no second encode
implementation to drift. The CLIP input stays wired so the contract and digest
can be read from it; reading a stamp does not load weights.

## Controls, before it is trusted

- Deliberate mismatch: the same graph with one key field changed
  (`qwen_short_edge` 0 to 2048, or v1 for v2) must miss. A cache that hits
  across that is the defect this note exists to prevent.
- Missing field: an entry written without a field, or a graph that cannot
  supply one, must refuse, not match.
- Round trip: a cached conditioning must be byte-identical to a fresh encode
  of the same inputs on the same artifact; if the encode is not
  deterministic on this kernel path, the note is wrong about what it caches
  and the control says so.
- The Sol sink: conditioning rows are in the exact sink, so a cached entry
  must reproduce the packed layout exactly (row counts, tags); the
  `check_typed_reference_consumers` snapshot is the model for that assertion.

## The vision tower, as the second and smaller item

The Qwen3-VL vision tower is identical across v1, v2 and BF16 and its output
for a given image depends only on the media bytes and the grid. Its seam is
one call, `comfy/text_encoders/llama.py`'s `self.visual(image, grid)`. A
per-image cache there pays only when the conditioning cache misses on a new
prompt that reuses a reference, and saves a fraction of one encode. Worth
doing after the conditioning cache, keyed on media sha256 and grid, never
before it.

## Not in scope

The DiT sees no benefit; nothing here changes sampling. The encoder v2
calibration is unaffected. Nothing here is a substitute for the ComfyUI node
cache in default mode, which already does the same-session job.
