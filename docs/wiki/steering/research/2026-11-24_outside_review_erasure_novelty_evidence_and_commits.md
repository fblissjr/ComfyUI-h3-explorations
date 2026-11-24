# Outside Review: Is Pole-A Erasure Novel, Does the Evidence Hold, and the 2026-11-20..23 Commits

**Date:** 2026-11-24
**Repo baseline:** `main` at `177004e7`
**Scope:** every commit outside `app/` from 2026-11-20 (the day `add_norm_matched_erasure` first landed, `6eb1f23a`) to the baseline, plus the steering docs, paper, records and code they produced.
**Companion pages:** [`../how_pole_a_erasure_works.md`](../how_pole_a_erasure_works.md) · [`../../h3_steering_and_architecture.md`](../../h3_steering_and_architecture.md) · [`../paper/manuscript.md`](../paper/manuscript.md) · [`2026-11-22_paper_v4_critical_review_prior_art_and_revision_plan.md`](2026-11-22_paper_v4_critical_review_prior_art_and_revision_plan.md) · [`2026-11-20_unpooled_multivector_sliders.md`](2026-11-20_unpooled_multivector_sliders.md)

This is a dated review, not a rule page. It was asked for by the owner as honest outside feedback on four questions: is the erasure concept and its math novel, is "multivector" valuable and untapped, what is being missed, overhyped or underhyped, and how do the commits of the window read. It changes no code and corrects no other page; section 8 lists what it would suggest, as advice only.

## How this was done, and how to read the status tags

Seven independent reviewers each took one lane (engine code, experiment records, two prior-art searches, the paper, the commit stream, the roadmap). They worked read-only and CPU-only, with no renders and no model loads, and were told not to trust prose. One further pass then re-read the load-bearing claims against the raw records and code.

Every finding below carries one tag:

- **[checked]**: re-read in that final pass against the cited record or code.
- **[reviewer]**: reported by one reviewer from its own scratch scripts, and not re-derived in the final pass.
- **[web]**: a source found by a reviewer's web search and not opened in the final pass. Treat titles, years and identifiers as leads to confirm.

This page carries no measured values. Each row names the record that holds them. The reviewers' full reports and scratch scripts are kept locally, untracked, under `internal/claude/2026-11-24_outside_review/`; they are not needed to follow this page.

## 1. Short answers

1. **Is erasure novel?** As an operator, no. As an exact recipe, it appears unpublished: two searches found every ingredient in print and no single work that combines them this way. The closest relatives are affine concept editing on the language-model side and soft-projection concept erasure on the diffusion side (section 3).
2. **Is the math behind it new?** No. The regression split offered as the reason erasure works is an identity that holds for any data, and its headline share appears just as strongly when Pole B is replaced by noise (section 4, row 6).
3. **Does it work?** One result is solid: `noir` flips to black and white at high strength on one prompt across seeds, and on clips. The records do not show that the concept-specific eraser matters: a blunt shrink of the prompt's tokens toward their own mean, which stores nothing, matches it wherever both were run (section 4, rows 1 to 5).
4. **Is multivector valuable and untapped?** Today it is mostly a label. The matched target direction is nearly the pooled one, the eraser reads Pole A only, and every push-side multi-vector extension came back null or negative on the repo's own gates (section 5).
5. **What is most durable?** Facts about this model rather than about the eraser: what visual steering does to audio, how early in denoising the look is set, and that reference images and keyframes lock the look whatever the text says (section 6).
6. **The commits?** Better guarded than their volume suggests, with a refactor that cannot be re-checked against the code it replaced, a body of stale doc anchors, and a decisions log that stopped a day early (section 7).

## 2. What ships, read from the code

| What the code does | Where | What the guides say | Status |
|---|---|---|---|
| The eraser acts on each selected token's deviation from the current prompt's own token mean. It shrinks token-to-token spread inside the Pole-A subspace and adds nothing to the prompt's average. A prompt that is uniformly Pole A in every token is not erased. | `steering/runtime.py::_erasure_row_shift64` | "subtract the part of the prompt that lies in Pole A's subspace" | does not hold as worded **[checked]** |
| The eraser's basis is the leading principal directions of the Pole-A span tokens, centered on one global mean. Pole B never enters it, and pair weights do not enter it. | `steering/extract.py::build_pole_a_erasure` | "centered per pair"; per-pair centering "dramatically outperforms" | the per-pair variant exists only in bench code (`bench/control_vector/eval_multivector_erasure.py`); the shipped builder is the global one **[checked]** |
| The matched target direction reduces to a similarity-weighted mean of Pole-B tokens minus the plain mean of Pole-A tokens. It is one vector, broadcast to every selected row exactly as the 1D vector is. | `steering/extract.py::build_pole_a_erasure` | "unpooled", "ColBERT-style", "multivector" | a mild reweighting of the pooled mean **[checked]** |
| At zero or negative strength the erasure is off and the direction is the 1D vector, so the map is not symmetric in strength. | `steering/runtime.py::_erasure_row_shift64` | one formula for all strengths | does not hold as written **[checked]** |
| The mode falls back to a plain push, with nothing logged, when fewer than two rows are selected, when the projection is near zero, when a stacked union is trivial, or when peer orthogonalization empties a slider's eraser. | `steering/runtime.py::_erasure_row_shift64`, `steering/runtime.py::_stacked_erasure_block_shift64` | "falls back cleanly" | holds, silently **[checked]** |
| An eraser never touches a peer slider's push. Pushes are summed as one constant row after the erasers act on the prompt's own variation. Peer orthogonalization changes only how much of the prompt's variation along a peer's target is removed. | `steering/runtime.py::_stacked_erasure_block_shift64` | "noir cannot erase cyberpunk's neon"; "zero peer-target erasure" | describes an overlap statistic, not a runtime effect **[checked]** by reading; a planted-overlap CPU test agrees **[reviewer]** |
| A peer at any positive strength, however small, triggers the full peer orthogonalization. | `steering/runtime.py::_stacked_erasure_block_shift64` | not mentioned | holds **[checked]** |
| A hard cosine gate decides whether the stacked cap applies, so two identical sliders and two nearly identical sliders get different total sizes. Exactly opposite pushes return the bare erasure; nearly opposite pushes return almost nothing. | `steering/runtime.py::_has_distinct_unit_peers`, `steering/runtime.py::_resolve_stacked_target_rms` | "smooth" | not continuous **[checked]** |
| In a capped stack, raising strengths further grows the push only slightly and shrinks the erasure. | `steering/runtime.py::_stacked_erasure_block_shift64` | not mentioned | measured on a real prompt **[reviewer]** |
| `sculpt` builds its union through a second implementation, so sculpting two sliders is not the same edit as chaining them. | `steering/extract.py::compose_erasure_payload` | presented as equivalent | differs **[reviewer]** |
| The bus-scoped eraser helpers are exported and tested but unused at runtime after their GPU gate rejected them. | `steering/extract.py::compute_bus_erasure_basis`, `steering/extract.py::project_erasure_orthogonal_to_basis` | n/a | dead code **[reviewer]** |
| The shadow text stream in the attention hooks relies on a core helper updating its buffer in place; the return value is discarded. | `steering/hooks.py::H3SteerRuntime` (the `_mod_gate` calls) | n/a | true today; fragile to a core change **[checked]** by reading |

A plain description an outside reader would recognize: an affine edit that adds a contrastive mean-difference vector and, in the same step, ridge-shrinks each token's deviation from the prompt mean along the leading principal directions of the baseline-pole tokens, then renormalizes the edit.

## 3. Prior art

Sources marked **[web]** come from the reviewers' searches; the others are works the final pass knows independently.

| Ingredient | Closest published work | How the shipped code differs |
|---|---|---|
| Add a contrastive mean-difference vector | Contrastive activation addition and difference-of-means steering. In generative media: text-encoder sliders on image and video models with language-model encoders (arXiv 2603.17998 **[web]**). On this model at this seam: the community pack compared in [`2026-11-06_mainodes_comparison_and_pika_roadmap.md`](2026-11-06_mainodes_comparison_and_pika_roadmap.md) **[checked]** | Token-span pooling and masked rows; otherwise the same seam |
| Remove the input's own component and add the target in one affine edit | Affine concept editing, "Refusal in LLMs is an Affine Function" (arXiv 2411.09003) | That work removes the single concept direction, with a hard projector, and re-centers on the negative-class dataset mean |
| A soft, ridge-weighted projector off a token set's principal subspace | Conceptor negation: word vectors (arXiv 1811.11001), contextual language-model embeddings (arXiv 2211.11087), steering with mean-centered conceptors and a Boolean OR that resembles the union rule (arXiv 2410.16314) | Truncated to a low rank, and centered on the prompt's own mean rather than a training mean. The steering docs never mention conceptors |
| The same projector family on diffusion text-token embeddings | CURE (arXiv 2505.12677); runtime projection of prompt tokens, including for video, in SAFREE (arXiv 2410.12761) | Those build the subspace from the concept to remove; here it comes from the baseline pole. They have no additive term and no norm lock. The docs cite CURE without noting the shared operator, and do not mention SAFREE |
| The affine family `W(x - mu_s) + mu_t` | Representation surgery / MiMiC (arXiv 2402.09631) | The shipped edit is a heuristic member: mean shift plus a covariance-shrinking map |
| Composing edits: orthogonalize against directions to keep, union the erasers | Retain-orthogonal steering (arXiv 2610.01969 **[web]**), conceptor OR, orthogonalized steering directions (arXiv 2602.15847 **[web]**) | A new arrangement of known parts; the eigenvalue clamp and the tanh cap are engineering choices |
| Token-level soft alignment to estimate the direction | Decomposable attention (arXiv 1606.01933) for the mechanism; the nearest steering use is arXiv 2603.02237 **[web]** | Apparently unclaimed as a steering-vector estimator, and small in effect here (section 5) |
| Routing steered text to a band of blocks | Extended textual conditioning (P+), block-targeted style methods, "vital layers" analyses of image transformers; on this model, the per-block text deltas in the 2026-11-06 comparison **[checked]** | Convergent with the community finding, not a first sighting |
| Different text views for audio and video | Separate text connectors per stream (arXiv 2601.03233 **[web]**); different prompts per stream at several passes per step (arXiv 2610.00691 **[web]**) | No work was found where, in one pass, audio queries read unsteered text while video reads steered text. This is the most defensible new piece, in an area that is moving fast |
| The regression split used as the mechanism argument | Linearity of ridge regression in the response; "regression of change on baseline", known as mathematical coupling | A textbook identity, not a result |
| Reading a vector back as words before rendering | Concept-decomposition and lens methods for diffusion conditioning | Apparently unclaimed as applied here; a diagnostic that the repo's own audit says does not predict renders |

**Apparently unclaimed after two searches:** centering on the current prompt's own token mean; the add-plus-erase-plus-norm-lock combination used as a creative slider on a joint audio-video model; the single-pass audio split. Confidence is moderate, and lower for late 2026, where search coverage is thin.

**Errors in the repo's own prior-art pages [reviewer]:** the critical review grades the audio-video routing as wholly new; it and the manuscript misdescribe several cited works (a lens paper that studies a vision transformer, a method that corrects the velocity output rather than text keys and values, one venue and year); two cited items could not be found at all; the v5 manuscript has no bibliography. The full table is in the local paper-audit report.

## 4. Evidence, claim by claim

| # | Claim as the guides word it | Record | What the record shows | Status |
|---|---|---|---|---|
| 1 | Erasure flips `noir` to black and white on stills where 1D stays color | `data/exp_unpooled_sliders/gate2b1/metrics_gate2b1.json` | Holds on every seed in the record, on one prompt, at the high-strength setting | holds **[checked]** |
| 2 | The same on clips | `data/exp_unpooled_sliders/gate2b2/metrics_gate2b2.json` | Erasure is less saturated than 1D on every seed: strongly on the first/last-frame family, modestly on the reference family. "Crisp black and white" was judged by eye | holds as "desaturated" **[checked]** |
| 3 | The effect is there at the node's default strength (`h3_rules.py::STEER_DEFAULTS`) | `data/exp_unpooled_sliders/phase3_2026_11_21/stage_b_surgical_erasure_v2.json`, `ramp_records` | Erasure does not beat 1D at the default; the flip appears only at the top step of the ramp. A single seed | does not hold at the default **[checked]** |
| 4 | Only the concept's own eraser works ("double dissociation") | `data/exp_unpooled_sliders/specificity_control/metrics_specificity.json`; the two gate records above | The swapped, random and prompt-direction arms ran on a single seed, and the record holds no color measure, so their labels were read off contact sheets. The blunt-shrink arm (`S4_iso_shrink`) matches the shipped eraser on `noir` on every seed, stills and clips | does not hold as worded **[checked]**. [`../../h3_steering_and_architecture.md`](../../h3_steering_and_architecture.md) states that arm failed |
| 5 | The eraser needs several dimensions | `data/exp_unpooled_sliders/paper_2026_11_21/gpu_ablations_1_to_5.json`, `ablation3_rank_and_ridge` | On `noir`, rank one flips as well as the shipped rank. The most heavily regularized ridge setting (the weakest eraser) does not flip, and in `ablation4_pair_count` neither do the low pair counts. A single seed | amount of shrink matters; rank does not, on this record **[checked]** |
| 6 | The regression split proves erasure is the working ingredient | `bench/verify_pole_a_erasure.py` | The identity holds for any data. Run with shuffled pairs, or with Pole B replaced by matched noise, the same statistic stays as high; with a pure additive shift as ground truth it falls to nothing. The reported figure is a cosine described as a share. The script's synthetic test builds Pole B independent of Pole A, so it passes by construction | not evidence of mechanism **[reviewer]**, found independently by two reviewers |
| 7 | The matched direction is required | `data/exp_unpooled_sliders/pre_integration/metrics_pre_int.json` | One concept on a few seeds separates it from the pooled direction; a second concept shows no difference. Across the bank the two directions are nearly parallel (`bench/control_vectors.py inspect <stem>`, field `cos(mu_d,v_1d)`) | holds narrowly; "required" cannot be told **[reviewer]** |
| 8 | Stacked erasure beats stacked 1D; peer orthogonalization fixed the lighthouse artifact | `stage_b_surgical_erasure_v2.json`, `stack3_records`; `data/exp_unpooled_sliders/stage1_2026_11_22/stage1_ros_norm_cliff.json` | The 1D stack is more desaturated than the erasure stack in most cells. The artifact measure barely moves when orthogonalization is switched on and drops only when the cap is lowered. On another suite in the same file orthogonalization does change saturation. One render per arm | mixed; the artifact fix tracks the cap **[checked]** |
| 9 | Erasure lowers flicker | the clip record in row 2 | The spread scales with the mean; relative spread is similar across arms | restates lower saturation **[reviewer]** |
| 10 | Erasure helps across concepts | `bench/results/2026-11-21_multivector_erasure_benchmark.md` and the records above | Of the concepts with a matched 1D-versus-erasure render, only `noir` and one utility stem show a win on a measure that fits the concept; the rest tie, show nothing, or cannot be judged by the recorded measure, and most rest on a single seed | breadth rests on `noir` **[reviewer]** |
| 11 | Blocks 20 to 29 are the style band | [`2026-11-21_phase3_de_slop_and_vision_block_steering.md`](../../../../bench/results/2026-11-21_phase3_de_slop_and_vision_block_steering.md), produced by `bench/control_vector/run_phase3_steering_experiments.py` | One concept, since retired; that band alone was never run; a later test found the wider band about equal to all blocks | cannot tell **[reviewer]** |
| 12 | Steering only the first half of the steps matches the full window | [`2026-11-22_phase4_clamped_stacking_and_audio_split_probes.md`](../../../../bench/results/2026-11-22_phase4_clamped_stacking_and_audio_split_probes.md), section 5a | Holds on the listed cells, two seeds | holds as recorded **[checked]** |
| 13 | Visual steering at the default streams setting is safe for audio | [`2026-11-23_wave2_gpu_gates_bus_vr_path1c_and_qvid.md`](../../../../bench/results/2026-11-23_wave2_gpu_gates_bus_vr_path1c_and_qvid.md), section 1.2 | The record itself notes pitch drift on speech when both streams are steered **[checked]**. A reviewer reads the same table as a loudness drop that the single-pass audio split mostly recovers **[reviewer]** | does not hold |
| 14 | Text steering can restyle a shot pinned by keyframes or a reference image | [`../how_pole_a_erasure_works.md`](../how_pole_a_erasure_works.md), section 4 | Repeated nulls, documented there as boundaries | does not hold, and the guides say so |
| 15 | Erasure beats rewriting the prompt | none | No such comparison exists for erasure. The nearest record, [`2026-11-12_new_vectors_three_ways.md`](2026-11-12_new_vectors_three_ways.md) section 11.1, is a playback cross-check that rated a plain rewrite and edit guidance clearly ahead of the vectors at their default settings | not tested **[checked]** |

**What the records support saying:** weakening the prompt's own token-to-token contrast lets a style vector overcome a prompt that describes the opposite, at high strength, shown robustly for one concept whose target is itself a removal. The concept-specific eraser may still preserve scene content better than a blunt shrink; that was judged by eye, on a concept that has since been retired.

**Two things outside the records [reviewer]:** the scripts that produced the 2026-11-20 control, specificity and gate records are not in the repo or its history; and a policy line in [`2026-11-22_track1_track2_multivector_erasure_blueprint.md`](2026-11-22_track1_track2_multivector_erasure_blueprint.md) (**[checked]**) ended 1D comparison arms from that date, so breadth evidence for erasure against 1D stops there.

## 5. What "multivector" means at each place it is used

| Use of the word | What it is in code | State | Recorded gain over the simpler thing |
|---|---|---|---|
| The matched target push (`erasure.mu_d`) | One vector; a reweighted pooled mean (section 2) | shipped, default | small and narrow (section 4, row 7) |
| The low-rank Pole-A eraser (`erasure.vr`, `erasure.diag`) | The one working part | shipped, default | large against 1D on `noir`; none shown against the blunt shrink (section 4, rows 4 and 5) |
| Stacked sliders | Peer orthogonalization, clamped union, soft cap | shipped | sound algebra, thin and mixed render evidence (section 4, row 8); the repo's own film record notes artifacts with two look sliders ([`2026-11-23_multishot_mixer_film.md`](../../../../bench/results/2026-11-23_multishot_mixer_film.md)) |
| Composed union in `sculpt` | A second union implementation | shipped | none measured **[reviewer]** |
| Explicit facet sub-vectors | planned only | no code | CPU estimates only **[reviewer]** |
| Implicit per-token routing | CPU prototype | falsified by its own gate | `data/exp_unpooled_sliders/phase5_cpu_audits/stage3_cpu_prototypes.json` **[reviewer]** |
| Vision-token patch vectors | code shipped, kept out of the bank | GPU null | the wave 2 record, section 2 **[reviewer]** |
| Bus-scoped erasers | helpers on disk | GPU rejected | the wave 2 record, section 1.1 **[reviewer]** |
| Word vectors with an `against` eraser | shipped | no effect on its one test **[reviewer]** | |

Everything multi-vector on the push side is near null on the repo's own records. The retrieval lineage (late interaction, token matching) is framing rather than mechanism; the repo's own critical review already reached that conclusion.

## 6. Ledger

**Overhyped**

- "Five independent proofs", "double dissociation", and the theorem, law and proof language in the paper. One proof is algebra; the others cite `noir` for every figure.
- Peer orthogonalization as what keeps sliders from damaging each other (section 2; section 4, row 8).
- The block band as a finding (section 4, row 11).
- The novelty of the audio-video routing as a whole. The single-pass audio split is the defensible part.
- The paper itself. A referee's read is that it is an internal engineering report today: nearly every GPU result is one seed on one scene, and several headline sentences go past their records **[reviewer]**.

**Underhyped**

- Visual steering changes the audio at the default streams setting, and the split that addresses it is off by default (section 4, row 13).
- The early-step window (section 4, row 12).
- The reference and keyframe lock, a reusable boundary for anyone using this model (section 4, row 14).
- The blunt shrink. If it holds up, a "prompt contrast" dial that stores nothing is a simpler tool than a per-concept payload.
- The engineering under the research: masked token spans, checkpoint-family fingerprinting, and guards that arrive with the code.

**Overlooked**

- A comparison against simply rewriting the prompt (section 4, row 15). For language models, a public benchmark found prompting ahead of activation steering (arXiv 2501.17148).
- Blind playback judgment by the owner of any erasure result in the window. Saturation is the main measure throughout, and a grayscale filter in post would pass it.
- The default strength sits below where the effect appears (section 4, row 3).
- Erasure became the node default the morning after it landed (`6eb1f23a` to `6d96b09a`), on evidence from essentially one concept.
- The stacking discontinuities and the silent fallbacks (section 2).
- Test coverage of the stacked path: several deliberate breaks of the stacked math pass every check, and two of the three steps in `bench/verify_pole_a_erasure.py` call no repo function **[reviewer]**.

**Sound as it stands**

- The `noir` result at high strength, stated with its prompt and strength.
- The reading that the earlier regression prototype was effectively an eraser.
- The keyframe and reference boundaries as documented.
- Node schema stability across the window: no renamed node, no input inserted mid-list, no zero sentinel, no owner paths **[reviewer]**.

## 7. The commits of 2026-11-20 to 2026-11-23

**What was good [reviewer]:** most code commits changed a guard in the same commit, and of the largest code commits nearly all came with a guard that exercises the new code. The CPU guards that could be run pass, apart from the two items below. The move into `steering/` reproduced the function bodies faithfully. Outside steering: origin-scoped keys with redirects off in the heylook client, and the Pika reference sizing change tied to an upstream change, logged and guarded.

**What to look at again:**

| Commit | Why |
|---|---|
| `db3d51c0` | Mixes the move into `steering/` with new behavior. Its parity check compares the new code to a formula retyped inside `bench/check_steering_v2.py`, not to outputs of the code it replaced **[reviewer]** |
| `340e8ded` | Hook seams on the attention path every render takes, re-implementing core block internals (section 2, last row) |
| `6d96b09a` | Made erasure the default one morning after it existed; every graph left on the default changed behavior |
| `cbbf565d` | Several GPU rejections, a paper closure and a default-model change in one commit, none in the decisions log **[reviewer]** |
| `92a9eac0`, `610c791b` | Features removed within days or hours of landing; the second also adds new behavior under a removal title **[reviewer]** |
| `14b8f8b3` | The third value of the heylook default model in under a day; worth confirming the final one is intended **[reviewer]** |
| `69498b35` | A large new check with no row in the checks index |
| `177004e7` | A truncation trim in pair generation with no check; `bench/control_vector/llm_pairs.py::_repair_multichange_markdown_pairs` drops the last pair whenever the text fails to parse **[checked]** by reading |
| `b42d443c` | The paper's build and audit code lives under `docs/`, outside the indexes and the guard sweep **[reviewer]** |

**Process observations:**

- **A guard is red at the baseline [checked].** `bench/check_heylook_prompts.py` and `bench/check_steering_v2.py` have no row in [`../../ops/checks.md`](../../ops/checks.md), which `bench/check_doc_inventory.py` exists to catch.
- **The decisions log stops at 2026-11-22 [checked].** [`../../decisions.md`](../../decisions.md) has no section for the 23rd, a day with a package move, a feature removal, several gate rejections and default changes **[reviewer]**.
- **Stale anchors [checked for one, reviewer for the count].** The move left many doc links whose label names `steering/` while the target is a line range in a file that is now a short re-export shim; [`../how_pole_a_erasure_works.md`](../how_pole_a_erasure_works.md) section 2a is an example. `bench/check_doc_links.py` does not range-check `#L` anchors, so these pass.
- **Two gaps in the sweep [reviewer].** `bench/run_checks.py` runs only `check_*.py`, so the `verify_*.py` scripts never run in it; and `bench/list_prose_measurements.py` skips inline code, so figures written in backticks are invisible to it.
- **Churn [reviewer].** Several features lasted hours: a cache added and removed the same afternoon, helpers rejected by their gate soon after landing and still on disk, node inputs that lived minutes. Many label series (wave, gate, phase, stage, track, path) ran at once, some reused for unrelated things, and superseded files do not say so in their headers.
- **Bug risks outside steering, read and not run.** `auto_edit_prompt.py` rejects an edit when any validation failure string is new compared with the original's (**[checked]** by reading). A reviewer adds that those strings embed lists, so an edit that partly repairs a broken original can be refused **[reviewer]**. `director/server.py::_probe_heylook_url` contacts the configured heylook host with its auth headers (**[checked]** by reading); a reviewer reports that the director check's offline mode reaches it **[reviewer]**.

## 8. Suggested order of work (advice, not a plan)

1. **Before more building, one fair comparison.** A handful of concepts (looks, materials, behaviors), a few prompts each including one with no conflict, several seeds. Arms: unsteered, 1D, shipped erasure, blunt shrink, and a prompt rewrite; at the default strength and at high strength; stills first, clips for the survivors; judged blind by the owner on playback, with the usual measures alongside as context. If the rewrite wins or ties wherever a rewrite is possible, erasure's role narrows to cases where the prompt cannot change. If the blunt shrink ties the shipped eraser, it can be the simple mode and the concept-specific claim is dropped.
2. **Rerun the specificity control** with every arm on several seeds and a color measure recorded, plus the rank sweep on several seeds.
3. **Test the case only erasure can serve:** same-prompt spatial portals and locked takes, against edit guidance with a rewritten edited prompt.
4. **Defaults worth a test:** audio protection and the early-step window for visual targets; the default strength against where the effect appears.
5. **Page corrections the records already support:** the blunt-shrink sentence in the architecture guide; the "share" wording for the regression statistic; "per-pair centering" where the shipped builder is described; the stale anchors into the shim files; a superseded banner on the v4 paper.
6. **Guard gaps:** rows for the two unlisted checks; the `verify_*.py` scripts inside the sweep; a stacked-path test that fails when the stacked math is broken.
7. **Ideas from the landscape, all training-free [web]:** gate the erasure per token so prompts with no conflict are left alone; a retain-aware projector in place of holds plus peer orthogonalization; per-token norm preservation in place of one fixed size; editing only text values in the band and leaving keys alone; a calibration that makes one strength mean the same amount across sliders; and an eraser built from Pole B for negative strength.

## 9. Limits of this review

- No renders and no GPU. Every statement about pictures or sound rests on existing records.
- One reviewer per lane. Where two lanes reached the same finding independently it is said so; otherwise a **[reviewer]** finding has one source.
- Prior-art sources from 2026 were found by search and not opened in the final pass. Search tools return summaries, so details can be wrong.
- The built vector files are not in the repo tree, so the bank-wide comparison of the matched and pooled directions was not recomputed in the final pass.
- Two slips by reviewers, for the record: one ran a single read-only `git status` against its brief, and one left a bytecode cache under `bench/__pycache__/`. No tracked file changed.

## 10. How to re-derive

- The gate, ramp, rank and stage records in section 4 are JSON; each row names its file and field.
- The regression identity and its figures: `bench/verify_pole_a_erasure.py --cpu-only` with ComfyUI's interpreter. The null controls are the same computation with the pairing shuffled, with Pole B replaced by Gaussian rows of matched covariance, and with Pole B set to Pole A plus one fixed vector.
- The matched-versus-pooled cosine for one stem: `bench/control_vectors.py inspect <stem> --prompt <scene.txt>`.
- The unlisted checks: `bench/check_doc_inventory.py`.
