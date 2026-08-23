# Conditioning against the release: what was built, and what was not

last updated: 2026-08-23

Two defects were found by reading the published release beside the code that
consumes it ([`official_weights_metadata.md`](official_weights_metadata.md)).
This records what was done about each, and — the part worth more — what was
deliberately *not* done and why, so nobody rebuilds the discarded option.

Source for the path itself: a full node-by-node trace of
`workflows/h3_text_to_video_turbo_768p_euler.json` against
`workflows/h3_image_ref_plus_text_to_video.json`, done 2026-08-21.

---

## The plan changed after the trace, and this is the change

The first sketch was **two replacement conditioning nodes**, one non-reference
and one reference, reimplementing what core does but correctly. The trace
refuted that shape. Two findings did it:

1. **`MiniMaxH3ReferenceToVideo` carries at least five load-bearing contracts
   that exist only in code comments** (listed below). A reimplementation drops
   one silently, and none of them raises when broken.
2. **Neither defect actually requires replacing that node.** The tokenizer one
   is fixed at the CLIP, one node upstream and one-tenth the surface. The pixel
   one does not bind on any graph this repo ships.

So the built thing is smaller than the sketch on purpose: fix the defect at the
narrowest seam that reaches it, and instrument the one that is not currently
reachable rather than patching speculatively.

That conclusion still governs the two defects above; it does **not** mean this
repo never needs a reference conditioner. A typed one was later built for a
different reason: make order, soundtrack ownership, loader metadata, and audio
normalization explicit. Native core's contracts remain tested, and its gaps
remain recorded. The local node handles selected gaps for this repo; it does
not turn them into native ComfyUI fixes.

---

## Defect 1: the seven special tokens. Fixed natively; local shim retained.

**Native ComfyUI status:** fixed by merged PR 15808, present in this installed
checkout as commit `924743af`. Core's H3 tokenizer now registers all seven
tokens itself. That is the resolution of the ComfyUI/vendor gap.

**This repo's handling:** `MiniMaxH3VendorTokens` and
`clip_with_vendor_tokens` predate the native fix. They remain only for older
ComfyUI installs and are required to be no-ops on this checkout. Removing their
workflow inputs is deferred in [`roadmap.md`](../roadmap.md) until the minimum
supported ComfyUI version is explicit.

`MiniMaxH3VendorTokens` (`vendor_tokens.py`) is the standalone CLIP-to-CLIP
form. The local conditioning nodes call the same module's
`clip_with_vendor_tokens` helper while their generated `vendor_tokens` input is
enabled. On an older unpatched install it reads the release's declared token
list from `vendor_config/` and adds what the bundled tokenizer lacks. This
describes the compatibility implementation, not the native fix.

The seam is a **rebind on a clone**, not a mutation. `clip.clone()`
(`comfy/sd.py:301-310`) copies seven fields and shares the tokenizer and the
text-encoder module by reference, so
editing the loaded tokenizer in place would reach every graph in the process —
the silent-contamination class `reference_fit.py` documents for its global
rebind. The node builds a fresh tokenizer instead and rebinds it on the clone,
which was verified: a second tokenizer constructed the same way is unaffected.

Three seams that look available and are not, all read from source so nobody
re-tries them:

- **`clip.set_tokenizer_option`** (`comfy/sd.py:315-316`) dead-ends. The option
  dict arrives at `MiniMaxH3Tokenizer.tokenize_with_weights`
  (`comfy/text_encoders/minimax.py:137-138`) as `**kwargs` and is never
  forwarded, so it cannot reach the inner tokenizer on this model.
- **The `..._tokenizer_class` hook in `tokenizer_data`** is consumed inside
  `SD1Tokenizer.__init__` (`comfy/sd1_clip.py:695`). A node running after
  `CLIPLoader` cannot reach it.
- **Setting the attribute on `clip.cond_stage_model`** (shared by `clone()` at
  `comfy/sd.py:304`) is process-global contamination for the same reason as
  above, not a scoped change.

**What this does not establish**, and the node's own docstring repeats it:
whether the embedding rows for these ids carry trained values, and what the
markers are for. The release lists them and documents neither.

## Defect 2: the pixel bounds. Detected, not patched.

The conditioner leaves `min_pixels` / `max_pixels` on a shared helper's
signature defaults where the release declares its own; the table is owned by
[`h3_references.md`](../h3_references.md).

**Neither bound binds on anything this repo ships.** `MiniMaxH3ReferenceFit`
puts every reference at a 2048 short edge, which sits above the release's floor
and below ComfyUI's ceiling until roughly 3:1. So the fix would be a monkeypatch
guarding a case that does not occur.

The trace also priced the two halves differently, which is why they are not
treated the same:

- **The floor is reachable from a node.** It only fires when the image is under
  the threshold, so pre-conforming the tensor makes the helper a no-op. That is
  `reference_fit.py`'s composition pattern one stage later. It costs forking the
  one-image-two-towers identity, at one extra resample.
- **The ceiling is not.** The clamp is on the helper's *output*, so no input can
  beat it — passing a larger image just gets it shrunk. It needs a scoped
  rebind, and the mechanism has two aspects nobody has verified: whether an
  already-loaded text encoder re-applies object patches, and whether an instance
  attribute holding an unbound function satisfies the call site.

**What was built instead:** `bench/preflight_graph.py` now warns, statically,
when a reference will cross either bound — naming which side would do what.
ComfyUI's values are read out of its source rather than restated, so the warning
goes stale loudly. When the source cannot be read it says the comparison was not
made rather than printing nothing.

That converts "should we write a monkeypatch?" from a guess into an observation:
build it the first time a graph trips the warning.

---

## The five contracts a replacement node would have had to reproduce

Recorded here because they are load-bearing, they fail silently, and every one
of them lives only in a code comment — `CLAUDE.md`'s "a requirement is not a
control" case, five times over.

**Five of the seven gained a control on 2026-08-22**, after standing
unenforced through two postmortems: `bench/check_reference_contracts.py`
asserts contracts 1, 2 and 3 plus both smaller ones against core's own
behaviour, driving the node with stub VAEs to the point where its two
reference lists are complete. Shown red by pairing soundtracks positionally in
core and confirmed green on restore.

**Contracts 4 and 5 gained controls later the same day**, closing the last two.
Both were held open by an assumption about where they could be asserted --
contract 4 is `model_base.py`'s job rather than this node's, and contract 5 was
recorded as needing a loaded encoder. Neither is true: `extra_conds` runs on a
stub whose only supplied attributes are `concat_keys` and `model_config`, and
`CLIP.encode_from_tokens` runs over a stub text encoder returning the same
3-tuple `MiniMaxH3ClipModel.encode_token_weights` returns. **The blocker was
where to point the harness, not what it would cost.**

Both are shown red in `bench/red/show_red_reference_contracts.py`, which
mutates the real functions in memory rather than on disk -- the install is
shared with a running render server. Three mutations, each phrased in the
contract's own terms: the concat order reversed, the `return_dict` merge
removed, and the tag copy deleted. A mutation whose anchor no longer matches
**refuses** instead of reporting a red, which is not hypothetical -- the first
run of that harness errored on a recompiled `super()` losing its `__class__`
cell, and the spine correctly scored it ERRORED rather than counting it.

1. **`ref_items` and `ref_blocks` are not the same length**
   (`comfy_extras/nodes_minimax_h3.py:290-342`).** A video with a
   soundtrack appends two entries to the presentation list and one to the DiT
   payload. Index-aligning them mislabels every reference after the first
   sounded video.
2. **The soundtrack pairs by socket-name suffix**
   (`comfy_extras/nodes_minimax_h3.py:313`), not by position:
   `ref_video_audio_N` belongs to `ref_video_N` through a string join. A
   mis-numbered socket silently pairs the wrong track.
3. **The `<Audio j>` counter is shared** across video soundtracks and standalone
   audio in one sequence, and standalone audio comes last. A prompt that says
   `<Audio 1>` means something different depending on whether a video soundtrack
   is wired.
4. **Keyframe latents precede reference latents** in the flat conditioning
   lists, established by statement order in `extra_conds` and nowhere else.
5. **`minimax_token_tags` reaches conditioning only through the
   `return_dict=True` path** (`comfy/sd.py:412-416`).** Nothing raises if it is missing; the DiT just tags
   every row as text.

Plus two smaller ones with the same property: the vision sentinels must flank
each block or two rows per block get tagged as text, and prompt weighting must
stay disabled or a CLIP-style blend is applied to a hidden state this model
never saw weighted.

---

## Controlled since 2026-08-22

The four defects this node exists to fix are asserted by
`bench/check_conditioning_behaviour.py`, against core's
`MiniMaxH3ImageToVideo` as the reference rather than against remembered
numbers. Both kinds of failure are covered: breaking something core got right
(the AGREE arms) and quietly ceasing to fix what it was built for (the DIFFER
arms). The second is the one nothing would otherwise notice, because the node
keeps running and the graphs keep rendering.

Shown red twice and independently: deleting the empty-prompt refusal reddens
only that arm, forcing `fit_to_canvas` reddens only the last-frame arm.

## Typed reference surface: built and migrated

The runtime replacement was added on 2026-08-23, after the acceptance suite
and ordered resolver had stopped changing under adversarial review:

- `MiniMaxH3AppendRefImage`, `MiniMaxH3AppendRefVideo`, and
  `MiniMaxH3AppendRefAudio` build one copy-on-append
  `MINIMAX_H3_REFERENCES` tuple;
- a video record owns its frames, the same loader's `VHS_VIDEOINFO`, and its
  optional soundtrack;
- `MiniMaxH3ReferenceConditioning` walks the tuple once to build Qwen's
  `ref_items` and the DiT's `minimax_refs` blocks;
- that boundary derives the source clock from `loaded_fps`, normalizes video
  to 24 fps, duplicates mono to stereo, and caps audio at aligned
  `frame_count / 24`;
- reference geometry remains Comfy-compatible and no-upscale. Release sizing
  is deliberately not mixed into an ordering migration.

`bench/check_reference_runtime.py` controls the runtime boundary with stub
VAEs and no CUDA; `bench/red/show_red_reference_runtime.py` makes ignored fps,
skipped audio normalization, foreign metadata, and reversed order go red.
`bench/check_typed_reference_consumers.py` proves label discovery and preflight
read the same validated chain.

**Live acceptance and workflow migration are green as of 2026-08-23.** All 38
shipped reference API graphs now use the typed conditioner; across UI, API, and
the stamped bench copy that is 77 typed files and no shipped core socket node.
The generator preserves the old image → sounded video → standalone-audio order
so existing prompts retain their ordinals, while hand-built typed chains may
use list order directly. Explicit audio trims were removed because the compiler
owns the aligned-duration cap.

The all-media acceptance graph passed the live schema and rendered at 1024x768,
39 frames, and 10 steps in 84.51 seconds. The first attempt found that VHS
returns a lazy `Mapping` rather than core's concrete audio dict; that boundary
was fixed and added to the CPU runtime check before the successful rerun. The
server logged `presentation=['<Picture 1>', '<Picture 2>', '<Audio 1>',
'<Video 1>', '<Audio 2>']`, 21,283 packed rows, Sage routing, Sol sparse
execution, and all twenty native tokens already present, so the local token
compatibility helper was a no-op.

Core's node is retained as a research subject and compatibility surface, not as
the shipped graph authority. Its still-open sizing, rate, duration, and mono
gaps stay documented as native gaps even where this repo's typed boundary
handles them.

The acceptance list above remains relevant: **since 2026-08-22 every item is
guarded by an assertion** in `bench/check_reference_contracts.py`, with
contracts 4 and 5 shown red in `bench/red/show_red_reference_contracts.py`.
This section said "not done" until the runtime existed; preserving that history
matters because the controls, not implementation enthusiasm, were the blocker.

**Read the acceptance criteria as two lists, not one.** Some of these are
behaviour a replacement must PRESERVE — a sounded video making two Qwen items
but one DiT block, the shared `<Audio j>` ordinal with a soundtrack
immediately before its video, the vision sentinels, weighting stayed disabled.
Others are behaviour a typed surface would INTENTIONALLY replace: pairing a
soundtrack by socket-name suffix becomes ownership by the video record, and
"standalone audio always comes last" becomes ordered-list position. A test
suite that asserts AGREE on all seven would reject the replacement for doing
its job, so the intentional ones have to be asserted as DIFFER.
