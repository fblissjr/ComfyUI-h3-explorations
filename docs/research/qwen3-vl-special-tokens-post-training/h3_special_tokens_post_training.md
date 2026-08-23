# MiniMax H3 special-token post-training: evidence and experiment design

Last updated: 2026-08-23. This is a research proposal, not a training recipe or
a released adapter.

## Short answer

The seven H3 markers are real tokenizer entries, but their Qwen embedding rows
were not post-trained by MiniMax. They occupy stock Qwen3-VL padding rows and
remain statistically indistinguishable from the unused tail of the embedding
table:

```
151669  <d>                 151673  <|lyrics_end|>
151670  </d>                151674  <|caption_start|>
151671  <|cutoff|>          151675  <|caption_end|>
151672  <|lyrics_start|>
```

That does **not** establish that the markers are useless. H3 consumes the
unnormalized output after Qwen language layer 50, and the frozen H3 DiT could
have learned to recognize the repeatable activation pattern produced by an
otherwise untrained row. Meaning can live in the downstream consumer even when
it is absent from Qwen's language-model objective.

It also does not establish which tokenization MiniMax used while training the
DiT. The public artifacts support at least two incompatible histories:

1. The DiT saw the dedicated ids. It may have learned their fixed layer-50
   patterns, so changing the rows now can reduce compatibility.
2. The DiT saw the ordinary BPE spellings used before the additions were
   loaded. In that case the release tokenizer is a train/serve mismatch, and
   assigning new semantics to the dedicated rows may still miss the training
   distribution.

Until that history is known or separated experimentally, training all seven
rows is not a safe general improvement. The smallest useful experiment is
`<d>`/`</d>` only, measured through the frozen H3 diffusion model on real
dialogue video/audio examples.

## What is established

The evidence already in this repo separates facts from hypotheses:

- [`official_weights_metadata.md`](../official_weights_metadata.md) records
  the release tokenizer, exact assigned ids, embedding-table shape, stock-Qwen
  control, and layer-50 encoder contract.
- [`2026-08-21_h3_token_embeddings.json`](../../../bench/results/2026-08-21_h3_token_embeddings.json)
  compares the seven rows against trained Qwen special tokens and unused
  padding rows. The official BF16 release and the local INT8 conversion agree;
  all seven sit at the untrained pole.
- [`2026-08-21_h3_marker_token_states.json`](../../../bench/results/2026-08-21_h3_marker_token_states.json)
  shows that dedicated ids, legacy BPE fragments, and deleted markers produce
  materially different layer-50 states. This is encoder evidence only; it does
  not say which representation the DiT prefers.
- H3 uses the first 50 of Qwen3-VL-32B's 64 language layers, returns the raw
  last-layer state without the final RMSNorm or LM head, and includes visual
  tokens and DeepStack features for image/reference prompts.
- The seven-token tokenizer gap is now fixed **natively in ComfyUI**, not by
  this repo. The installed core adds them in `MiniMaxH3Tokenizer`; the local
  compatibility shim is a no-op on patched installs.

One fact that is *not* established is the tokenizer configuration used for
MiniMax's DiT training. Untrained rows prove that Qwen did not learn them. They
do not prove that the frozen encoder never emitted them while the DiT trained.

## Why the original prototype was removed

The first draft paired a sound high-level warning with code that violated it.
The code and synthetic JSONL samples were removed rather than published as a
runnable recipe.

| Prototype behavior | Why it was invalid |
|---|---|
| Optimized causal next-token cross entropy | No language-model loss observes H3 video/audio fidelity or the frozen DiT's conditioning contract. |
| Imported `Qwen2_5_VLForConditionalGeneration` | The artifact is Qwen3-VL, and H3 consumes a different 50-layer, no-final-norm path. |
| Claimed a contrastive dataset | The trainer ignored the positive/contrast relationship, and the contrastive JSONL did not contain the `prompt` key it required. |
| Used text containing `<Picture 1>` without image tensors | This did not exercise Qwen3-VL vision preprocessing, visual tokens, MRoPE, or DeepStack. |
| Put the full embedding table in AdamW | That creates optimizer state for about 778 million values to tune 35,840 of them. With decoupled weight decay, rows with masked zero gradients can still change. |
| Saved a full replacement embedding table | The result would be roughly 1.5 GiB in BF16, lacked source hashes and sparse-row metadata, and was not loadable by the current ComfyUI node as an overlay. |
| Suggested batch four on one GPU | A BF16 Qwen3-VL-32B training forward plus optimizer state cannot fit a 24 GiB RTX 4090. |
| Treated generated prompt strings as training data | Diffusion grounding requires licensed target video/audio (or a justified teacher signal), exact H3 preprocessing, and reproducible noise/timestep records. |
| Included caption examples | The caption-token semantics are not documented in either official prompt guide and have not been behaviorally established here. |

The 100-row samples also had very low combinatorial diversity: the
"contrastive" file contained only 61 unique complete rows, and its alleged
negatives changed both marker syntax and the semantic description. Such pairs
cannot isolate a delimiter effect.

## The hypotheses an experiment must separate

### H1 — dedicated-id training contract

MiniMax loaded `additional_special_tokens` while training the DiT. Qwen stayed
frozen, but each dedicated id produced a stable contextual layer-50 signature
which the DiT learned as a boundary code.

Prediction: the release-id arm beats BPE-fragment and stripped-marker arms on
dialogue boundary and synchronization metrics. Moving the rows without a
frozen-DiT objective is likely harmful.

### H2 — BPE training contract

MiniMax's training tokenizer did not realize the additions, so the DiT learned
the ordinary pieces for strings such as `<d>` and `</d>`. The public serving
tokenizer later began emitting the dedicated ids.

Prediction: the legacy-BPE arm beats the release-id arm. A possible repair is
not generic language-model training; it is either restoring the training-time
tokenization or learning dedicated rows that reproduce useful BPE-conditioned
behavior.

### H3 — markers are mostly redundant

Nearby prose such as “says”, the language label, punctuation, and audio context
carry the task. The exact delimiters have little marginal effect.

Prediction: dedicated-id, BPE, and stripped arms overlap across seeds and
held-out prompts. In that case post-training adds risk without evidence of a
benefit.

The first experiment should choose among H1–H3 before optimizing any row.

## Required baseline: no training

Run three tokenizer arms on identical weights:

1. **Release ids:** `<d>` and `</d>` become 151669 and 151670.
2. **Legacy BPE:** reconstruct the unpatched tokenizer and emit its fragments.
3. **Stripped:** remove the markers while leaving the dialogue text intact.

Use common latent noise and timestep draws for loss-level measurements. For
generation-level measurements, use a seed distribution rather than reading a
single divergent sample as a quality ranking. Keep the text encoder, DiT, VAE,
reference media, prompt prose, scheduler, resolution, duration, and sampler
identical.

The repo already has the encoder-level version of this control in
`bench/grade_h3_marker_tokens.py`. The missing part is a DiT-boundary or
distributional render evaluation.

## If H1 wins: do not post-train Qwen first

The dedicated rows are already the codebook the DiT learned. An attractive
language-model embedding is irrelevant if it produces the wrong layer-50
pattern for that codebook.

Only consider tuning if a concrete H3 failure remains after release-id
tokenization is verified. Optimize through the frozen H3 objective, train a
sparse row delta, and make the baseline row values recoverable exactly.

## If H2 wins: two repair directions

The lowest-risk repair is to use the legacy BPE tokenization that the DiT
appears to prefer. It changes no model weights.

A more experimental repair is **representation transplantation**: train only
the dedicated rows so the frozen Qwen layer-50 states approximate the useful
legacy-BPE arm across many contexts. This is encoder distillation, not semantic
grounding. Because one dedicated token replaces two or three BPE pieces and
changes later positions, exact equivalence is impossible; the objective must
align shared ordinary-token spans and report the residual rather than promise
a perfect transplant.

Even a successful transplant must still pass the frozen-DiT evaluation. Lower
layer-50 error is only a proxy.

## The defensible training parameterization

Do not make the full embedding table trainable and mask its gradient. Keep the
base table frozen and substitute a small parameter of shape `[N, 5120]` only
at selected ids:

```
base embedding lookup (frozen)
        |
replace selected positions with base_row + trainable_delta[N, 5120]
        |
frozen Qwen layers 0..49, no final norm
        |
frozen H3 DiT loss and optional representation constraints
```

For the first pilot, `N=2`, so only 10,240 BF16/FP32 values plus optimizer
state are trainable. Save a sparse overlay containing:

- token string and resolved id;
- base-row hash and full source-checkpoint hash;
- tokenizer/config hashes;
- delta tensor, dtype, and training commit;
- objective weights and data-manifest hash.

The loader must reject an overlay when any provenance value disagrees. It must
also have a literal off state that reproduces the base encoder bit-for-bit.

## Objective

For a real H3 training example with target video/audio latents, noise sample
`epsilon`, timestep `t`, conditioning `C(delta)`, and frozen DiT `D`:

```
L = L_h3_diffusion(D, target, epsilon, t, C(delta))
  + lambda_context * L_context
  + lambda_radius  * L_delta_radius
```

`L_h3_diffusion` must match H3's actual joint video/audio training parameter
target and weighting. That implementation is not present in this repo today;
an inference sampler is not automatically a faithful training loss.

`L_context` is measured on aligned, non-marker token states **inside prompts
that contain the markers**. Marker-free prompts are already exactly invariant
when only sparse marker rows are substituted, so a marker-free anchor loss adds
no information. A “strictly unchanged” context constraint is also impossible:
the point of modifying a marker row is to alter contextualized states. Use a
declared deviation budget instead.

`L_delta_radius` prevents an unconstrained row from escaping far outside the
embedding distribution. Its scale must be selected on validation behavior,
not from the norm of the untrained tail alone.

## Data required

Synthetic prompt strings are useful for tokenizer and structural tests, but
they are not diffusion post-training data. Every train/eval row needs source
media rights and a manifest containing mode, prompt, target clip, reference
items, preprocessing policy, duration/fps, and hashes.

### Pilot: `<d>` and `</d>`

Use diverse, accurately aligned clips covering:

- one and multiple visible speakers, including off-screen speech;
- silence, silent reading, narration, and non-speaking mouth motion as hard
  controls;
- several languages, accents, ages, vocal timbres, speaking rates, and line
  lengths;
- dialogue at the beginning/end of a clip and multiple separated spans;
- camera cuts during speech, occluded mouths, profiles, and wide shots;
- T2VA, I2VA/FL2VA, and Ref2VA prompt structures;
- ordinary marker-free H3 prompts for regression evaluation;
- empty, unmatched, nested, and adjacent-marker examples for rejection tests,
  not as positive training rows.

Record word/phoneme timing, speaker visibility, active-speaker identity, and
voice-activity intervals where licensing and annotation permit it.

### Later markers

- Lyrics require paired vocal/instrumental material, aligned lyric spans,
  speech-vs-song controls, multiple genres/languages, and singing-specific
  evaluation. A 300 Hz–3 kHz energy heuristic is not a singing detector.
- Cutoff requires precisely timed incomplete utterances near the generated
  endpoint plus naturally completed controls. The official prose spells this
  `<cutoff>` while the tokenizer declares `<|cutoff|>`; the contract must be
  settled first.
- Caption markers should remain out of training until their intended behavior
  is established. Neither official guide documents them.

## Evaluation and stop conditions

Pre-register held-out prompts and report all arms, not only the proposed
overlay:

- frozen H3 validation loss using common noise/timestep samples;
- release-id vs legacy-BPE vs stripped baselines;
- same-seed multi-seed generations with blind labels;
- lip-sync and active-speaker metrics, ASR/WER, voice activity boundaries, and
  human ratings for dialogue;
- aligned layer-50 relative L2/cosine on ordinary tokens in marker-containing
  prompts;
- exact bitwise equality on marker-free prompts with the overlay enabled;
- malformed-marker behavior and long/multilingual prompt stress cases;
- memory, encoder latency, and render latency.

Stop if the baseline does not separate H1/H2/H3, if gains do not reproduce
across seeds and held-out prompt families, or if the overlay degrades the
release-id baseline outside the target task. Do not expand from dialogue to all
seven tokens merely because the training loss goes down.

## Practical status

No valid post-training implementation exists in this repo yet. The next useful
work is a no-training DiT-level three-arm dialogue characterization. If that
shows a stable dedicated-id deficit and identifies the preferred target
representation, build the two-row sparse overlay and its off-state/invariance
harness before acquiring a larger dataset.
