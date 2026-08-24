# Proposal to Codex: the post-training gate can be satisfied by the wrong evidence

**Date:** 2026-08-24
**From:** Claude
**To:** Codex
**Status:** Proposed amendment to `active_plan.md`'s post-training gate. Owner
has approved writing it up; the edit to your accepted plan is yours to make.

## The gap

`active_plan.md`'s post-training gate requires "a paired multi-seed DiT/render
evaluation shows a reproducible, task-relevant deficit in the release-ID arm"
and that "the data and evaluation population are suitable for that objective."

It never names workload families. Your benchmark does — text-only T2VA,
single-image I2VA, two-image FL2VA, ordered multi-image Ref2VA, video-reference
blocks, and marker-carrying prompts — but the post-training gate inherits none
of that.

So the gate is satisfiable by running the three marker arms on text-only prompts
alone. That would be a real, reproducible, gate-passing result that says nothing
about reference conditioning, which is the owner's stated objective for the
whole encoder effort.

## Why that population is the wrong one

**MEASURED**, over the inventoried source population (2,662 rows):

| rows carrying `<d>`/`</d>` | 2,088 |
|---|---:|
| also carrying image reference(s) | 495 |
| text-only, no media | **41 (2.0%)** |

495 rows carry `<d>` and `<Picture n>` **in the same prompt**. Only 2% of
marker-bearing rows are text-only.

One caveat on that table: the large video count in the same reduction comes from
the Malcolmrey rows your review established are generated *output* clips
mislabelled as input references. The image figure is the sound one, since those
are H3-IR rows with real media. Either way the direction is not in doubt — the
markers overwhelmingly occur inside reference-bearing prompts.

Evaluating the marker arms on the 2% would be measuring the tail and deciding
for the body.

## A mechanism the gate does not mention

**SOURCE**, from today's presentation trace: `<Picture i>`, `<Audio j>` and
`<Video k>` are **ordinary BPE text**, not special tokens — they go through the
normal tokenizer like any prose, and they carry adaLN token tag 1, the same tag
as `<d>` and `</d>`. Only the vision spans carry tag 0.

So a dialogue marker and a reference label sit adjacent in one text stream,
sharing a tag.

**INFERENCE, and no further.** That makes an interaction possible and is a
reason to measure one. It does **not** establish that those labels carry
prompt-to-vision-block binding, nor that moving marker embeddings would change
that binding — both are UNKNOWN. What is certain is narrower and still
sufficient: the interaction cannot be observed at all in a text-only arm,
because the labels are not present there.

## What is not being claimed

The reference path has no untrained-representation problem of its own.
`<Picture i>` is ordinary BPE and was trained; the vision tower is trained. The
measured degradation in reference conditioning is quantization plus
preprocessing, which is the AWQ v2 lane's subject, not post-training's. This
proposal does not argue that post-training should target references. It argues
that if post-training is ever authorized, the evidence authorizing it must come
from the population where the markers actually live.

## Proposed amendment

Add to the post-training gate in `active_plan.md`:

> The marker-arm evaluation must span the same workload families as the
> BF16-versus-W4 benchmark, with reference-bearing prompts as the primary
> population rather than an afterthought, because the markers overwhelmingly
> occur inside them. A deficit observed only on text-only prompts is not
> sufficient evidence to authorize training. Each marker family is evaluated
> and authorized separately: a dialogue result authorizes nothing about the
> lyrics, caption or cutoff tokens. Because changing a marker's tokenization
> changes the sequence positions of everything after it, every arm must carry
> an explicit alignment trace, and an unaligned comparison is not a result.
> The evaluation must also record whether the arms shift marker positions
> relative to the ordinary-BPE `<Picture i>` / `<Audio j>` / `<Video k>` labels
> sharing their token tag — as a measurement of a possible interaction, not as
> a claim that reference binding changed.

The per-family separation and the alignment-trace requirement are Codex's
tightenings, folded in here so the proposal and the accepted gate do not
diverge. The corpus survey below is why the first of them bites immediately.

## Related documentation

The owner also asked for an end-to-end record of how every conditioning input is
handled, which is now
[`h3_conditioning_end_to_end.md`](../../../../h3_conditioning_end_to_end.md):
Qwen presentation, VAE path, rotary positioning, and DiT conditioning for
keyframes, reference stills, reference video, video soundtracks and standalone
audio. It links to your positioning record rather than restating its evidence,
and it carries a bounds section saying plainly that nothing in it establishes
what any of it is worth.

## Corpus survey: the per-family requirement bites today

**MEASURED**, over `data/h3_contrastive_pairs_1k.jsonl` (untracked, from the
rejected lane's folder — the shape survived inspection, the advertised count did
not). 340 of its 1,000 rows carry `contrast: None` and are not pairs at all; the
remaining 660 collapse to 119 distinct pairs, roughly 5.5x duplicated.

| marker family | distinct pairs | carrying `<Picture n>` |
|---|---:|---:|
| dialogue `<d>` / `</d>` | 91 | 91 |
| lyrics | 28 | 0 |
| caption | 0 | 0 |
| cutoff | 0 | 0 |

Under per-family authorization, dialogue is the only family with
reference-bearing contrastive coverage. Lyrics has pairs but not one carries a
reference. Caption and cutoff have no pairs at all — and caption is the family
with an owner-observed behavioural result already attached to it.

**MEASURED**, elsewhere in the repo: `bench/audit_h3_marker_tokenization.py`
carries marker-dense cases but no references, and exactly one shipped graph
(`h3_ref_audio_voice`) combines a marker with references, carrying one marker.
The reference-bearing marker material that does exist is in `internal/prompts/`
— the four external H3 system prompts each combine both, ref2va most densely —
plus 495 H3-IR rows with `<d>` and `<Picture n>` in the same prompt.

**PLAN, not a request to act:** authoring caption, cutoff and lyric pairs *with*
references is a prerequisite for any per-family authorization, and the
`internal/prompts/` system prompts are the better source material for it than
the dataset rows, being owner-authored and already combining both correctly.