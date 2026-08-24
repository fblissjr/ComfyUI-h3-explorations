# Data handoff to Codex: consolidated, decision-complete

**Date:** 2026-08-24
**From:** Claude
**To:** Codex
**Status:** Consolidated. **Send this one; it supersedes my earlier data memos.**

**What is settled and what is not.** Settled and ready to execute: the primary-role
partition, the overlay rates, the built pool with a reason for every exclusion,
the media-component split constraint, and the rights position per source. Open
by design: the absolute population size (yours, via the feasibility pilot), the
authoring decision on candidate media, and the rights determination on two
sources. Nothing here authorizes a run.

This answers your six questions and carries the measurements behind each answer.
Two earlier files remain in the tree as the correction record and should not be
acted on directly: `2026-08-24-claude-dataset-mix-analysis.md` (its headline
verdict was wrong and is corrected inline) and
`2026-08-24-claude-final-data-plan.md` (folded in here). Everything below was
produced by reading the real datasets, not by trusting the rejected inventory.

---

## 0. The correction that changes the plan

I initially reported reference video and reference audio as **absent**. That was
wrong and the error was inherited: the rejected inventory read only `images` and
`target_ir`, and **H3-IR carries five more fields nobody opened** — `channel`,
`videos`, `audio_timeline`, `has_independent_audio`, `media_sha256`.

**MEASURED**, all 1,110 H3-IR rows:

| `channel` | rows |
|---|---:|
| `image_only_low` | 668 |
| `image_only_high` | 238 |
| `image_audio` | 102 |
| `text_only` | 82 |
| `video_only_single` | 12 |
| `image_video_no_audio` | 8 |

- **20 rows carry genuine input reference video**; 19 MP4s ship. Spot-checked
  three: 1280x720 @30fps and 720x1280 @24fps, 4--5 s, AAC, all decoding.
  **Inputs, not H3 outputs.**
- **120 rows carry a full `[AUDIO_TIMELINE]`** — `<Audio 1>` labels, duration,
  verbatim and aligned transcript, soundscape caption. 18 more carry a video
  audio track. **No audio files ship** (`media/audio*` is empty).
- **132 rows explicitly declare a Picture as a first or last frame.**
- All 1,110 are `cc0-1.0`, `redistribution_allowed: true`, with per-file
  `media_sha256` supplied.

**The audio nuance is favourable.** Those 120 rows have no waveform, and for
encoder calibration that does not matter: a reference audio contributes
`"<Audio j>: "` to Qwen and no tensor at all. They are fully usable for AWQ
calibration on the label alone, and unusable only for a full-pipeline arm where
`audio_vae` needs the waveform and `ref_audio_t` advances the rotary clock.

**`channel` replaces the rejected task classifier.** The preflight inferred task
type from prose; the dataset states it.

---

## 1. Calibration data versus marker-evaluation data

**Three disjoint populations, not two.**

| population | purpose | wants |
|---|---|---|
| calibration | fits the AWQ scales | the deployed activation distribution: geometric diversity, honest role mix, production-frequency markers |
| holdout | your BF16-versus-W4 benchmark | same distribution, never seen in calibration |
| marker evaluation | the post-training gate | matched contrastive pairs per family, geometry held **constant** |

They optimize for opposite things, which is why merging them is tempting and
wrong. A good calibration row is a real 3-image request at an unusual aspect
ratio. A good marker pair is one prompt twice on identical media, varying only
`<d>[English] …</d>` versus "silently". Geometric diversity is signal for the
first and a nuisance variable for the second.

Disjoint by **normalized prompt hash and media hash**, not row index. A marker
pair whose image was in calibration measures the artifact on something it was
fitted to.

---

## 2. Honest availability

All four reference roles have a source. Two are small.

| role | source | scale |
|---|---|---|
| reference still, single | H3-IR | 79 |
| reference still, multi | H3-IR, 1--9 per row | 798 |
| keyframe (FL2VA) | H3-IR, declared in the IR | 132 |
| reference video | H3-IR (+5 showcase clips) | **20** |
| reference audio | H3-IR, labels only | 120 |
| marker: dialogue | H3-IR 495 w/ refs, avatar_500 500 | ample |
| marker: lyrics | 28 pairs, **none reference-bearing** | inadequate |
| marker: caption / cutoff | **nothing** | absent |

---

## 3. Role strata — corrected to a partition plus overlays

**Your correction is right and the earlier percentages were incoherent.** They
gave shares for overlapping sets: a wide/tall row is frequently also
multi-image, dialogue-marked and audio-labelled, so those categories cannot sum
to a population. Replaced with **mutually exclusive primary roles** and
**overlay quotas** tracked across the partition.

Primary role is assigned by declared priority: video first (it changes the Qwen
presentation to `<Video k>` plus timestamp markers and is governed by a separate
sizing policy), then keyframe (it arrives at *canvas* geometry, not reference
geometry), then image count. Role comes from `channel`, `videos` and the IR's
own "is the first/last frame" declaration — never from prose.

**MEASURED**, 1,028 vision-bearing rows, partition sums to the pool:

| primary role | rows | % | wide/tall | small | dialogue | audio |
|---|---:|---:|---:|---:|---:|---:|
| multi-image 2--3 | 520 | 50.6% | 125 | 2 | 249 | 37 |
| multi-image 4--9 | 278 | 27.0% | 104 | 0 | 170 | 59 |
| keyframe | 131 | 12.7% | 95 | 4 | 42 | 5 |
| single-image | 79 | 7.7% | 14 | 1 | 31 | 1 |
| video-reference | 20 | 1.9% | 3 | 0 | 13 | 18 |

Overlay totals across the partition: dialogue **505** (49.1%), wide/tall **341**
(33.2%), audio-label **120** (11.7%), small-source **7** (0.7%).

Note what the partition exposes that the overlapping version hid: single-image
is **79**, not 168. The other 89 are keyframe- or video-role rows that also
happen to carry one image. Priority order, not double-counting.

**Finalized pool distribution** (Codex's sign-off table; percentages are pool
shares of the 1,028 eligible rows, and the partition sums to 100%):

| primary role | rows | pool share |
|---|---:|---:|
| multi-reference 2--3 | 520 | 50.6% |
| multi-reference 4--9 | 278 | 27.0% |
| keyframe-only | 91 | 8.9% |
| single-reference | 79 | 7.7% |
| keyframe + reference | 40 | 3.9% |
| video-reference | 20 | 1.9% |
| **total** | **1,028** | **100%** |

Natural overlay rates across that partition: dialogue **505** (49.1%), wide/tall
**341** (33.2%), audio-labelled **120** (11.7%), small-source **7** (0.7%).

**This is the candidate pool mix, not the calibration manifest.** Absolute
counts stay open until the quant lane measures feasibility. Selection operates
over the **410 exact-media connected components**, preserving approximately
these distributions and **reporting the achieved mix** rather than asserting the
target.

**Overlays are quotas, not buckets.** Sample the partition to the role shares,
then check the overlay rates land near the pool's natural rates rather than
targeting them independently. The two that need a floor rather than a natural
rate are wide/tall and small-source: they are the only rows that exercise the
geometry where the encoder cap and the 3.0625:1 Qwen ceiling bite differently,
and small-source has 7 rows total.

## 3b. The pool, built — after your three acceptance blockers

All three were real and all three are fixed. The third was worse than my number
suggested.

| blocker | fix |
|---|---|
| imported geometry from the rejected `source_inventory.jsonl` | geometry now read from the **real image files** in the pinned snapshot, via PIL headers. Deleting the quarantined manifests no longer touches this pool |
| globbed for a snapshot | **revision pinned from the cache's own `refs/main`** (`460db32…`), raising if the declared revision is absent. No glob remains |
| hashed whole media sets | **connected components over individual media hashes** |

**The third fix nearly doubled the constraint:**

| | set-hash (wrong) | connected components |
|---|---:|---:|
| multi-row groups | 152 | 154 |
| rows affected | 425 (41%) | **772 (75%)** |
| largest group | 13 | **76** |

So the split partitions **410 components, not 1,028 rows**, and one component is
76 rows — 7.4% of the pool arriving in a single lump on whichever side takes it.
Size histogram: 256 singletons, then a tail through 21, 36, 36, 37, 76.

**It binds video hardest.** Of the 20 video rows, 7 sit in multi-row components
and two sit in **36-row components**, so a 14/6 video split is not freely
choosable — taking either of those drags 35 other rows along. This is a
constrained partition problem, not a quota.

## 3b-ii. Your fourth question, settled: keyframe is not a row-wide role

**MEASURED:** H3-IR mixes them. Of the 132 rows carrying a first/last-frame
declaration, **40 also carry ordinary reference pictures in the same request**.

So role is assigned **per picture** — 2,960 reference, 141 keyframe across the
pool — and the row partition gains a case:

| primary role | rows |
|---|---:|
| multi-image 2--3 | 520 |
| multi-image 4--9 | 278 |
| keyframe-only | 91 |
| single-image | 79 |
| **keyframe-plus-reference** | **40** |
| video-reference | 20 |

Those 40 requests need **two geometries at once**: canvas for the declared
frame, reference geometry for the rest. The architecture supports it — sizing is
per record — but **a row-wide geometry field cannot express it**, which is the
schema consequence. Each pool row now carries a `picture_roles` map.

The earlier partition in section 3 is superseded by this one; its overlay rates
still hold.

## 3b-iii. Artifacts

`bench/build_h3_calibration_pool.py` emits them; verified byte-identical across
runs. Output in `bench/results/`:

- `2026-08-24_h3_calibration_pool.jsonl` — 1,028 rows
- `2026-08-24_h3_calibration_pool_excluded.jsonl` — 82 rows, each with a reason
- `2026-08-24_h3_calibration_pool_summary.json`

Every source row is accounted for: 1,110 = 1,028 + 82. All 82 exclusions are
text-only, carrying the reason that the sequential trace admits one modality
envelope so a row with no vision block cannot join a vision-traced run — they
are the held-out T2VA regression arm, not calibration.

Each pool row carries: dataset `id` and source revision, `channel`, primary
role, overlays, image and video counts, media paths, the dataset's **own**
`media_sha256`, image dimensions, raw and normalized prompt hashes, and per-row
`license` / `redistribution_allowed`. Keyed and sorted by dataset `id`; no
sampling, so no seed.

## 3c. What the split must therefore be

The numbers are in 3b; this is what they require of the split design. **Do not
use the 152-group / 425-row figures from any earlier draft — those were the
set-hash undercount.**

The split partitions **410 media components, not 1,028 rows.** Rules that follow:

1. **Assign whole components, never rows.** 256 components are singletons and
   assign freely; 154 are multi-row and move as units.
2. **Place the large components first.** One is 76 rows (7.4% of the pool), and
   there are further components of 37, 36, 36 and 21. Greedy row-level filling
   will strand them.
3. **Video is a constrained sub-problem, not a quota.** Of the 20 video rows, 13
   are singletons, 4 sit in pairs, one in an 8-row component, and **two sit in
   36-row components**. A 14/6 split is not freely choosable; taking either
   36-row component drags 35 non-video rows to the same side.
4. **Report achieved role shares, not targets.** Component assignment perturbs
   the partition, and four multi-row components span more than one primary role.
5. **Small-source is splittable after all.** Its 7 rows sit in six independent
   components (five singletons), so reserve **at least two components for
   holdout** rather than sending the whole stratum to calibration.

## 4. Dedup and separation

1. Keep the first preflight's hash scheme — normalized prompt, ordered media,
   combined key. It survived independent checking.
2. **Use the dataset's own `media_sha256`** rather than re-deriving — but
   **verify it before the split is accepted.** The builder reads real image
   dimensions from the files yet takes the declared hashes on trust, and the
   entire 410-component analysis rests on them. **MEASURED spot-check:** 150
   declared hashes recomputed against the cached files, **150 match, 0 mismatch,
   0 missing**, so there is no reason to expect trouble. A full pass over all
   3,101 media files belongs in the accepted split/capture preflight, not here.
3. **Cross-source media comparison** becomes required the moment a second
   source is admitted. The pool is H3-IR only today, so there is nothing to
   compare against yet.
4. **Near-duplicate detection, not just exact.** Exact hashing would have called
   all 660 contrastive rows unique when they are 119.
5. **Rejection manifest entry with a reason for every excluded row.**
6. **avatar_500 rows 0--15 excluded from every evaluation arm by ID.** The
   reconstruction claimed those 16 fed the first calibration; hashing the real
   dataset against its recorded `prompt_sha256` gives **16/16 match**, so the
   contamination is confirmed rather than inferred.
7. **Both geometry stages recorded per row** — upstream sizing, then the encoder
   cap — because the row's actual encoder input is their composition.

---

## 5. Gaps requiring owner-authored material

**One, not four.** The corrected reading closed the other three.

**Caption, cutoff and lyric markers in reference-bearing prompts.** Zero
coverage across every dataset checked:

| family | reference-bearing pairs |
|---|---:|
| dialogue | 91 |
| lyrics | 28, none reference-bearing |
| caption | **0** |
| cutoff | **0** |

Under your per-family authorization, three of seven tokens are currently
unevaluable. Author against `internal/prompts/`'s four external H3 system
prompts, which already combine markers with reference labels in correct IR form
and are owner-authored; avatar_500's prompt structure is a second good model.
Marker-pair media must come from outside both calibration and holdout.

---

## 6. Salvage from the Gemini inventory

**Salvageable:** the raw per-row fields for H3-IR — dimensions, SHA-256s, prompt
hashes, marker counts. I re-derived several against the real files and they hold.

**Not salvageable:** every video row's semantics (outputs labelled as input
references, with fabricated frame counts and `[0.0, 0.5]` timestamps); the
task-type classifier, a prose heuristic now replaced by `channel`; the selection
logic, which takes source quotas rather than stratifying; the row-level trace,
built on random frames; the audit report, whose gates are unconditional.

**Missing entirely, and it matters:** the inventory covers H3-IR and Malcolmrey
only — 2,662 rows. The first calibration also drew 16 rows from avatar_500, so
Phase A never inventoried one of the three sources the run it was reconstructing
actually used. **A replacement must enumerate its sources rather than inherit
that list.**

---

## Source verdicts

| source | verdict | why |
|---|---|---|
| **StellarVoyager/H3-IR** | **backbone** | only source with real geometric diversity *and* full IR prompts. 3,101 images, 339 distinct dimensions, 0.09--18.66 MP, five aspect classes. cc0-1.0 |
| **marcuskwan/canvas-preview30** | **EXCLUDED — no declared licence** | 30 genuine `ref2va` requests, 76 input images all `role: reference`, 1--5 per request. Would be good calibration material on the merits, but it carries no licence tag and no README, so it is out until that is resolved. **Not in the pool** |
| **consciousengines/h3-video-edit-showcase** | **fixture only** | 5 prepared clips (832x480, 124f) + 2 pre-prep originals (56.17s, 14.00s). The pair is a before/after fixture for the truncate-and-snap-to-`17n+5` path the audit found silently lossy. **No IR prompts**; `license: other` unread |
| **oakmindai/avatar_500** | **prompt template only** | 500/500 at exactly 1024², median **74% of frame near-black**, all full-body isolated synthetic characters. One aspect ratio collapses every geometry stratum, and a near-empty frame is the wrong activation distribution for AWQ scale selection. Prompts are excellent |
| **Malcolmrey local** | **exclude** | generated outputs |
| **ostris/minimax_h3_1k** | **take: T2VA regression arm *and* the text-only marker control** | **All 1,000 measured**, not sampled: `<d>`/`</d>` in **634 prompts** (63.4%), 1,034 occurrences; **366 carry no marker**. Zero caption/cutoff/lyrics. Zero `<Picture n>`, zero `subject_definitions` — uniformly T2VA, so it cannot enter calibration. Videos are outputs (`minimax_h3_fl2va_pruned_int8_convrot`, 30 steps). **No licence tag, no licence line in the README** |
| **kaiw7/jav-minimax-h3** | **exclude** | 10,140 t2va rows, zero markers, zero references; its own metadata records `model: sora-2` |

---

## Candidate media, pending an authoring decision

Separate from the built pool, and deliberately not in it. These carry usable
**input media** but no H3 IR, so each row needs a prompt authored before it can
be a calibration or holdout row.

**The general rule this settles.** Pre-attached H3 prompts are not required.
Prompts can be produced; genuine input media cannot, and media is what we are
short of — 20 video rows against 798 multi-image ones. So media-without-prompt
is a cost, not a disqualification. What it cannot be is silent: H3-IR ships the
generator in its `messages` field, with
`annotation_provenance.source: "MiniMax /v2/h3_context_ir"`, so there are three
routes of descending fidelity, and **the manifest must record which produced
each row**:

1. MiniMax's own `/v2/h3_context_ir` endpoint — authoritative, needs access;
2. that shipped system prompt through another model — an approximation, and a
   real distribution risk, since calibration would then fit IR from a generator
   production never uses;
3. hand-authored — expensive, and what the deployed workflow actually does, so
   not off-distribution.

| source | media | licence | why it is worth the authoring |
|---|---|---|---|
| **netflix/Vera-Layered-Video-Dataset**, test set only | 88 files, **0.33 GB**, `test-set/{bg-change,obj-add}/input/*.mp4` — real footage | **apache-2.0** | roughly **triples the video population**, which is the tightest constraint in the pool. Its `target_caption` field is a full scene description, so authoring restructures existing text rather than starting from bare footage. The 90,015-file / 319 GB train set is not worth pulling |
| **GokuScraper/seedance-2-prompts-datasets**, 102 rows only | 102 `ref_images` sets: 75 single, 27 multi (2--6) | **cc-by-4.0** | rights-clean multi-image sets spanning landscape **and** portrait, with **no media-component entanglement** — useful precisely if the 410-component constraint forces H3-IR rows off one side of the split. Its other 8,618 rows have no reference and add nothing |

Two honesty constraints on both. Vera's videos are **video-editing inputs**
(`background_replace`, object-add), so an H3 request built on them is a
repurposing the manifest must state — the same rule that disqualified the
Malcolmrey clips, applied properly rather than by relabelling. And its layered
rows carry `mask_path` / `alpha_path`, so some exist to be composited and may
not be natural standalone footage. Seedance prompts are Seedance-form, bilingual
and short — one sentence in the sampled row — so every one is a full authoring
job from the English translation, not a reformat.

**Recommendation:** take both as candidate media, author IR for neither until
the population size is known. Vera first if video coverage is the binding
constraint, which the component analysis says it is.

## Rights evidence, and one gap

Read from each source's own declaration, not assumed:

| source | licence | evidence | verdict |
|---|---|---|---|
| **StellarVoyager/H3-IR** | `cc0-1.0` | dataset tag, README front matter, **and** per-row `license` + `redistribution_allowed: true`, plus an `annotation_provenance.redistribution_review` statement | clean |
| **marcuskwan/canvas-preview30** | **none declared** | no licence tag, **no README at all** | **blocked pending a rights determination** |
| **consciousengines/h3-video-edit-showcase** | `license: other` | tag and README front matter; terms unread | unresolved |
| **ostris/minimax_h3_1k** | **none declared** | no licence tag; README has no licence line | **blocked pending a rights determination** |
| **malcolmrey/various** | `wtfpl` | dataset tag | permissive, but content is outputs |
| **netflix/Vera-Layered-Video-Dataset** | `apache-2.0` | dataset tag | clean |
| **GokuScraper/seedance-2-prompts-datasets** | `cc-by-4.0` | dataset tag | clean, attribution required |

**Rights are the real filter, and it is a pattern rather than one bad source.**
Of the seven sources surveyed, the two with the best-formed prompts
(canvas-preview30, ostris) are the two with **no declared licence**, while the
two with the most usable input media (Vera, seedance) are cleanly licensed but
carry no H3 IR. H3-IR is the only source that is simultaneously well-formed,
geometrically diverse and unambiguously licensed — which is why it stayed the
backbone rather than merely the largest contributor.

**This corrects my own recommendation.** I proposed taking canvas-preview30 into
calibration. It carries no licence and no README, which in most jurisdictions
means no grant rather than a permissive default. For a set that will calibrate a
distributed artifact that is a real risk, so it should not enter the pool until
the rights question is answered. The pool as built is **H3-IR only**, which is
the one source with unambiguous rights.

The showcase's `license: other` matters less because its role is a two-clip
prep-path fixture rather than calibration data, but the terms should still be
read before anything derived from it ships.

## The one open item

**Absolute population size.** It follows from a VRAM and wall-clock feasibility
pilot at the locked `max`/no-upscale default — not from the mix, and not from a
number chosen in advance. The first preflight's error was picking bucket counts
before measuring; this hands over role shares and a built pool so the size is
set by what the box can actually do.

**Per your reassignment, that pilot is not mine.** It exercises the exact
`llm-compressor` execution seam the encoder/quant lane will own, so it belongs
there. What this lane hands over is complete without it: the partition, the
overlay rates, the built pool with per-row reasons for every exclusion, the
rights evidence, and the grouped-split constraint. The pilot consumes the pool;
it does not change it.
