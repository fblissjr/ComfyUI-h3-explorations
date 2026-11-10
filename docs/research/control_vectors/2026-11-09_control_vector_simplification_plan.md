# 2026-11-09: control vectors, a plan to simplify, measure and extend

A dated record. It proposes; nothing in it is built. It was written from a read of `main` at `856d4ac8`
(code, the pair files, the wiki pages and one results record) plus CPU-only runs of the pair compiler.
No GPU build and no render was run for it, and each finding below says how it was checked.

The control-vector code was being changed by another session while this was written, so every finding here
is a snapshot of `856d4ac8`. Section 13 says what to check again before acting on it.

Related: [`2026-11-06_j_lens_and_refiner_jacobian.md`](2026-11-06_j_lens_and_refiner_jacobian.md),
[`../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md),
[`../../wiki/control_vectors.md`](../../wiki/control_vectors.md).

## 0. Where things stand (read this first)

This page grew by additions through one day: findings, a plan, a reorder, and corrections after a second
session read it. This section is the consolidated state and wins where an older section differs. Sections 1
to 16 are the reasoning and the evidence.

State at the last update of this section: the encoder change and this plan are on `main` at `402c074a`; the
bank is rebuilt (both families of every pair file, under the Pika recipe); the encode pass of section 17.4 is
running. The log from `402c074a` to `HEAD` says what moved since.

### 0.1 Order of work

1. The agreed fixes of 0.4.
2. The fork: edit guidance and prompt scheduling under Pika, at 141 frames, each across the whole schedule
   and with the edit switched on after the first steps, beside the plain prompt, the rewritten prompt and a
   random vector of the same size (sections 11.6 and 15.2).
3. If the fork goes well, the edit-guidance node (section 11.8). Then the apply node in its agreed shape
   (0.2), with the aimed mask (section 11.4).
4. The direction track (section 8).
5. Pairs last, from the existing files only. The scene pool and the pair-authoring prompt stay parked.
6. The CLI is cut down to its target shape (section 19) once step 2 has said whether the single-prompt
   vector method survives, since that decides one of its commands.
7. The wiki pages are rewritten (section 18) after steps 1 to 6, since they were rewritten before the code
   they describe changed. Two false statements in them are corrected at once, with step 1.
8. The Studio UI is left for another day. Nothing here maintains it (section 19.4).

Running beside these, and blocking none of them: the J-Lens decision (section 17.4).

### 0.2 Agreed by both sessions

- Pair-free methods are tried before any pair work; writing pairs is de-emphasized (section 12).
- The image graphs default to the 32B encoder, at the owner's word; the 4B encoder is a probe.
- In-place edits with several clauses are pooled in one pass over all changed spans. The single-clause
  string check stays for pair files.
- One apply node on the model wire, with the vector picked by concept name and an optional vector socket.
  Inputs with no remaining use are dropped.
- The scene mask becomes a real conditioning key, and a missing mask is a refusal.
- The bank mean is not subtracted by default. A direction built from neutral pairs that differ only in
  length is the test for a length effect.
- Edit guidance gives both passes the same position origin and keeps the two prompts from being batched on
  the image model.
- J-Lens is a readout first. A builder for the refiner half of its bundle is tracked from `7c01c792`; the
  lexical half is still copied from the existing file (section 17).
- Candidate directions are ranked by what the DiT does with them, not by the refiner alone.
- The first test runs at 141 frames.

### 0.3 Not settled

| Question | The two readings | What settles it |
|---|---|---|
| Why do the bank's vectors share a component? | That they do is now agreed: both sessions ran the command of section 15.3 on the finished rebuild, in both families and in both spaces. Two causes are on the table. The steered clause is longer than the plain one in most files. And the newer files share one kind of plain side (ordinary, natural, realistic), so every vector subtracts much the same thing. | Build the neutral-length direction of section 15.2, and a direction from pairs that vary only the plain side, and see how much of the shared part each explains. |
| Does pooling the unchanged rows steer better? | Measured (section 20.2): it is a different direction from the default, much smaller, and consistent from pair to pair. Whether it steers better is not something the rows can say. | One render arm at matched size, when a render round has room. The default does not change on this. |
| Do the reference-gate inputs stay? | One reply said remove them, a later one that they unlock late stylization. The record has one arm, which supports the second. | The owner's call when the apply node is rebuilt; a repeat of that arm if it matters. |
| Can `amplify` on a word's rows stand in for attention emphasis? | Offered as equivalent by the second session. It is not the same operation (section 15.4). | A small test, labelled as a stand-in, if section 11.3 is wanted before an attention hook exists. |

### 0.4 Defects both sessions agree on, to fix

| Defect | Where |
|---|---|
| `completely_preserved` taught as a retention marker; the release's closed set is `fully_preserved`, `partially_preserved`, `attribute_transfer`, `weak_reference` | `docs/wiki/system_prompts/h3_control_vector_system_prompt.md` (the example and the list), `docs/wiki/control_vectors/examples_and_recipes.md` (the advice and the worked prompt); all from `7c01c792` |
| The wiki says holds use `torch.linalg.qr`; the code is a two-pass Gram-Schmidt loop | `docs/wiki/control_vectors/building_pairs_and_vectors.md` against `control_vectors.py::hold_basis` |
| An image prompt compiles to `frame="raw"`, which the validator rejects | finding 2 |
| The Pair 0 splice, its silent no-op and its truncation of the pair list | finding 3 |
| A pole that occurs twice is replaced once in silence; `;` inside prose splits a clause | finding 4 |
| The span code looks the tokenizer up under one hardcoded name | finding 14; it now matters for the two `te4b` probes |
| The mask is attached inside a bare `except`, and a missing mask falls back in silence | finding 13 |
| A single pair reports a consistency of 1.0; the image node repeats one pair over canned scenes | finding 7 |
| Three canned ref2va scenes in node code | finding 17; section 16 |
| Three different defaults for the pair generator's output token limit | section 15.5 |
| A `hold` node whose mask differs from the add node's mask does nothing, with no message | `control_vector_nodes.py::_make_control_vector_wrapper` groups nodes by mask before holds are applied; shown by the option test of section 20.3. To become a refusal |
| `hold_words` with no vector and no words is accepted by validation and then ignored | `control_vector_nodes.py::attach_control_vector` returns before using them. To become a refusal |
| The build falls back to no LoRA in silence when the default Pika LoRA does not resolve | `bench/control_vector/build_cmd.py` (the `except` after `_parse_lora_specs`), and the same in `bench/control_vector/j_lens_builder.py` |

## 1. The plan in one picture

One data model, one build path and one small apply node. Every method becomes a way of producing pairs or
a vector, and each default is chosen by a measurement before the bank is rebuilt.

```
concept (tracked) + scene pool (swappable)
        | author: in-place on a prompt, an LLM edit per scene, image pairs
        v
     pairs (generated, under data/)
        | build: one path, Pika recipe, prints a held-out score
        v
     vector --> load / words / combine nodes --> apply node --> render
                     ^
     direction catalog (section 8) feeds the same socket
```

## 2. What the owner decided on 2026-11-09

- Pika H3 Flex is the default recipe. Vectors and the J-Lens bundle are built under it.
- A vector built from one prompt is a worked example, not tracked bank content.
- Breaking saved graphs is fine: `workflows/build_workflows.py` regenerates them.
- The wiki pages are rewritten from scratch after the code settles, by a reader who did not build it.
- The Studio UI waits until CLI to inference works end to end.
- The two P-H3-Image graphs are experimental and lower priority than video.
- Finding directions without prompts, pairs or words (section 8) is part of the plan.

## 3. Findings the plan rests on

"Run" means the compiler was executed on CPU. "Read" means the code was read and not executed.

| # | Finding | Where to look | Checked by |
|---|---|---|---|
| 1 | Two description clauses plus a soundscape clause in one spec are refused; one description clause plus one soundscape clause compiles | `control_vectors.py::differing_span` | run |
| 2 | An image prompt compiles to `frame="raw"`, which the metadata validator rejects | `h3_pair_compiler.py::compile_words_to_pairset`, `control_vectors.py::VALID_FRAMES`, `_validate_metadata_dict` | compile run; the save refusal was read, not run |
| 3 | A pair file given with `--target-prompt` gets a spliced "Pair 0", returns unchanged in silence when the clause is absent, and keeps only `pairs[:9]` | the Pair 0 splice helper in `bench/control_vector/helpers.py`, removed in `3f35901e` | read |
| 4 | A pole that occurs twice is replaced at its first occurrence only; a `;` inside prose splits the clause; with several scenes the substring must be in every one | `h3_pair_compiler.py::_apply_clause_inplace`, `_split_clause_statements` | run |
| 5 | Every text method pools only the rows whose wording changed, then adds one vector to every masked row | `h3_spans.py::pooling_mask`, `control_vectors.py::apply_post_refiner` | read |
| 6 | Holds are one mean difference per group, orthogonalized; the wiki calls it an SVD basis | `control_vectors.py::hold_basis` | read |
| 7 | One pair reports a consistency of 1.0; the image node repeats one image pair over canned scenes so the score is defined | `control_vectors.py::quality_report`, and the canned scene list in `control_vector_nodes.py`, removed in `3f35901e` | read |
| 8 | In nearly every pair file the steered clause is longer than the plain one, so vectors may share a component | load each file with `control_vectors.load_pairs` and compare `a_span` and `b_span` lengths | run; the shared component itself is not measured |
| 9 | J-Lens word directions are the output head times a Jacobian, the direction that makes the encoder predict a word next; its exponent and negative weights were fitted against the bank | `j_lens.py::synthesize_from_words`, section 6.1 of the J-Lens note | read |
| 10 | No tracked file writes the J-Lens bundle | `git grep -l J_49_to_63` returns readers and docs only | run |
| 11 | The apply hook is placed correctly: core refines text once per sampling, so the wrapper sees post-refiner rows and only edits `context` | `<comfy>/comfy/model_base.py` (`extra_conds`), `control_vector_nodes.py::_make_control_vector_wrapper` | read |
| 12 | The apply node does five jobs: load a vector, synthesize from words, hold words, an audio/video split that runs the DiT twice per step, and gating reference latents | `h3_rules.py::CONTROL_VECTOR_DEFAULTS`, `CONTROL_VECTOR_OPTIONAL_DEFAULTS` | read |
| 13 | The scene mask rides on a Python attribute of the tags tensor, set inside `except Exception: continue`; when it is absent `scene_only` falls back to all text rows with a log line | `h3_spans.py::attach_span_masks`, `control_vector_nodes.py::_resolve_custom_inference_mask` | read |
| 14 | The span code looks the tokenizer up as `qwen3vl_32b`; the image encoder registers as `qwen3vl_4b`, so the image graphs get no mask | `h3_spans.py::attach_span_masks`, `<comfy>/comfy/text_encoders/minimax.py::PH3ImageTokenizer` | read |
| 15 | The image graphs apply a vector built from 32B rows under the Pika recipe to rows from the 4B encoder with no Pika LoRA | `workflows/image/h3_p_h3_image_*_control_vector_api.json` | read |
| 16 | Graphs name a vector by pooling and LoRA hash, and a resolver swaps siblings by filename | `workflows/build_workflows.py` (the `control_vector=` entries), `control_vector_nodes.py::resolve_vector_file_path` | read |
| 17 | Hardcoded prose remains: three canned ref2va scenes that assert a person, the J-Lens word tables, a filler sentence in grading | the canned scene list in `control_vector_nodes.py` (removed in `3f35901e`), `j_lens.py::AGAINST_PRESETS` and the stopword tables, `bench/control_vector/grading.py::_build_padding_prompt` | read |
| 18 | The generator reads its system prompt out of the wiki page, and that prompt still teaches the retired slot compiler | `bench/control_vector/llm_pairs.py::_extract_system_prompt_text`, [`../../wiki/system_prompts/h3_control_vector_system_prompt.md`](../../wiki/system_prompts/h3_control_vector_system_prompt.md) | read |

What holds and needs no change: all pair files under `prompt_bank/control_vectors/` load, are t2va format, and
use distinct scenes and distinct wording per pair. The results record has render arms for t2va-built vectors on
ref2va, one seed each.

## 4. The methods we end with

All four end in the same kind of vector and the same apply node. The first three produce pairs and share one
build path.

| Method | You supply | Good for | Main limit |
|---|---|---|---|
| In-place on your prompt | one prompt and `old wording -> new wording` | a strength dial on a change to one shot, at unchanged prompt length | tied to that scene; its real competitor is rewriting the prompt |
| Concept over a scene pool | a concept and a set of scenes; an LLM writes the edit for each | reusable vectors; the only method with a held-out score | needs an LLM pass and a varied pool |
| Image pairs | several matched before and after pictures, plus your prompt | looks that are hard to put in words | unproven here; no render record was found for it |
| Words (J-Lens) | a few single words | instant sketches with no encoder load | lowest fidelity; single words only |

Beside these: combining existing vectors (add, invert, hold out), a reference image as a prefix on both poles
for ref2va, and the J-Lens readout of any vector into words. Section 8 adds a fifth source, and section 11
lists ways to steer that use no vector at all.

## 5. Data model

- The concept is the tracked thing: the `[vector]` header a pair file already carries (`name`, `quality`,
  `a_side`, `b_side`).
- The scene pool is a set of real prompts in all three formats, swappable. A starting pool can be the scenes
  already inside the pair files plus the `prompt_bank/` test prompts, minus what is held out.
- The pairs are generated output under `data/control_vectors/`, regenerable, never the definition.
- If every concept is built over the same pool, concepts differ only in their edits, which makes the shared
  component in finding 8 measurable.

The pair-file schema stays loadable as it is. Optional additions, each only when a measurement asks for it:
a scene id per pair (to hold out or regenerate by name), a mode per pair (today `mode` is per file), image
fields per pair, and a provenance table naming the model, system prompt and pool.

## 6. The system prompt

Today the prompt asks a model to invent ten scenes and write each twice, identical except one clause.
`bench/control_vector/llm_pairs.py::_repair_pair_prompts` exists because models fail at that.

The proposed task is smaller: give the model one pool scene and the concept, and ask only for the edit, the
exact clause to replace and its replacement. The in-place compiler applies it.

- Pairs are identical outside the edit by construction, so the repair code can go.
- The output is one line per scene.
- Variety and format coverage become properties of the pool, checked by code.
- The vendor-guide rules leave the prompt, since the model no longer writes scenes.
- The LLM method and the in-place method become one mechanism with a different author of the edit.

What stays in the prompt: one contiguous clause plus optionally one soundscape clause; never touch dialogue,
tags or headers; keep the concept's words out of the plain side; fresh wording per scene. The prompt becomes a
data file that the wiki links to.

## 7. Phases

The one ordering that matters: phase 2 comes before phases 3 and 5, so the bank is rebuilt once.

### Phase 1: make it run end to end

- Stop the image path writing `frame="raw"` (finding 2).
- Decide the multi-clause rule (finding 1). Suggested: one pair per clause, averaged.
- Remove the Pair 0 splice and its truncation (finding 3).
- Turn the silent edges into refusals (findings 4 and 13).
- Fix the hardcoded tokenizer name (finding 14).

Gate: one vector builds from the CLI and renders in the text-to-video and ref2va graphs.

### Phase 2: measure, then choose

One encode pass over the pair files, cached. Everything after it is CPU.

- A held-out pair score per concept: build from all pairs but one, score the one left out.
- The cosine between unrelated concepts (finding 8).
- Pooling the changed rows against pooling the unchanged rows, under the current `scene_only` mask (finding 5).
- t2va-format pairs against ref2va-format pairs for the same concept.

Gate: a table that picks the pooling default and says whether to subtract a bank mean. No renders.

### Phase 3: one build path

- Merge `bench/control_vector/build_cmd.py` and `bench/control_vector/make_cmds.py` into one build function.
- The CLI becomes `pairs`, `build`, `inspect`, `grade` and `lens`; each method's objective and limits go in
  its help text and in a comment beside its code.
- Apply the data model of section 5 and the prompt change of section 6.
- Remove the canned media scenes, the 1.0 scores for a single pair, and the prompt-guessing word tables from
  the build path (findings 7 and 17).
- Add a tracked builder for the J-Lens bundle under the Pika recipe (finding 10).
- Trim the C7 block of `bench/check_control_vectors.py` and move its prompt text into fixture files.

### Phase 4: one small apply node

- A control-vector socket fed by load, words and combine nodes. Apply keeps the vector, strength, mask and
  step window.
- The reference-latent gate becomes its own node.
- The scene mask becomes a real conditioning key.
- Graphs name a concept; the loader matches the build to the loaded recipe by metadata (finding 16).
- Change the generator entries and rebuild: one text-to-video graph, one ref2va graph, the stacked graph
  optional, the image graphs marked experimental.

Gate: checks green, and the list in section 9 verified by a reader who did not build it.

### Phase 5: rebuild and prove

- Rebuild the bank with the phase 2 defaults.
- One render round that decides: each arm with a repeat and a second shot, judged by the owner on playback.
- Two control arms in that round: a random vector of the same size as the steered one (how much does any
  large shift change the shot?), which the results record does not have, and the prompt simply rewritten
  (the thing every vector competes with), which it has for some arms (section 15).
- Then the image comparison: the same vector in the 32B-encoder graph and the 4B-encoder graph (finding 15).

### Phase 6: write it down, then the UI

- Rewrite the wiki pages from scratch. A capability is listed only with a link to the arm that showed it;
  everything else is listed as untested.
- Then the Studio UI, on the simplified CLI.

## 8. Direction discovery track

Today `j_lens.py::refiner_singular_vectors` takes the strongest directions of the refiner Jacobian and labels
each with lens words. The table in section 3.1 of the J-Lens note shows the labels are muddy. Two reasons:
it finds what the refiner amplifies, not what the DiT responds to, and the single-word readout is the weakest
labeller we have.

This track starts after phase 2 and runs beside phases 3 to 5. Nothing in the main plan waits on it.

- D1, candidates (cheap, mostly CPU): the refiner's strongest directions as now; the main directions real
  prompts vary along, from the phase 2 cached rows; a sparse dictionary trained on those rows, if the pool
  gives enough rows.
- D2, rank by what the DiT does: add each candidate at a fixed size and measure how far the predicted
  velocity moves on captured latents, split into video and audio, under two or three masks. Forward passes
  only, so it works with the quantized checkpoints, and `grade` already measures this kind of movement.
  Gate: a ranked table. No renders.
- D3, label the survivors three ways: the lens words, the tokens in real prompts that align most with the
  direction, and the existing concept vectors it overlaps.
- D4, render the top few, each with a repeat and a second shot. A direction enters the catalog only if it
  does something nameable both times.
- D5, use the catalog: entries load through the phase 4 socket; a pair-built vector can be decomposed against
  it to show what else it carries; a hold can name a catalog axis.

| Step | Depends on |
|---|---|
| D1 | phase 2 cached rows, phase 3 bundle builder |
| D2 | D1, the `grade` tooling |
| D3, D4 | D2 |
| D5 | phase 4 socket |

## 9. Both ends of the apply-node change

Phase 4 changes a shared node's inputs. Found by grep on `main` at `856d4ac8`; a build re-derives this list.

- Writers of the node's inputs: `workflows/build_workflows.py` (`_normalize_control_vector_specs`, the
  `control_vector=` entries, `build_control_vector_arm`, `build_control_vector_gate_arms`,
  `build_control_vector_probe_arm`), and the defaults in `h3_rules.py`.
- Readers: `control_vector_nodes.py::MiniMaxH3ApplyControlVectors` and `attach_control_vector`,
  `bench/control_vector/grading.py::_attach_raw_vector`, the five control-vector graphs under `workflows/`
  and `workflows/image/`, `bench/node_id_manifest.json`, and `bench/check_control_vectors.py`.
- The scene mask key: written by `conditioning.py`, `reference_conditioning.py` and `reference_encode.py`
  through `h3_spans.py::attach_span_masks`; read by the wrapper in `control_vector_nodes.py`.

## 10. Not verified, and other ideas

- Whether the tensor attribute carrying the scene mask survives `MiniMaxH3Preflight` in practice (finding 13).
- Whether the shared component in finding 8 exists; phase 2 measures it.
- Whether single directions in section 8 will be nameable. The existing table suggests many will not be.
- The outcome of the bare-frame comparison planned as E2.5 in
  [`2026-11-04_h3_control_vectors.md`](2026-11-04_h3_control_vectors.md): this page first said no result
  was found. One exists, in a file this read had not opened (section 15).
- The formulas in the wiki were not checked line by line beyond what section 3 lists.
- Later ideas that block nothing: a J-Lens word dictionary built from how rows look when a word is present,
  and steering the picture rows for image-pair vectors.

## 11. Steering without a vector

Added the same day. Everything above works by adding one vector to many text rows, which is a blunt tool.
The methods here use no vector and no pair bank. None has been tried in this repo: no H3 graph under
`workflows/` uses a guidance node (`grep -l '"CFGGuider"' workflows/*.json workflows/*/*.json` is empty), so
each starts with a small test.

### 11.1 Edit guidance: steer with the edited prompt itself

Each step, run the DiT on the prompt and on the edited prompt, and move the prediction from the first toward
the second by a scale. That is the ordinary guidance formula
(`<comfy>/comfy/samplers.py::cfg_function`: `uncond_pred + (cond_pred - uncond_pred) * cond_scale`), so core's
`CFGGuider` (`<comfy>/comfy_extras/nodes_custom_sampler.py`) may do it with no new code: the edited prompt as
positive, the original as negative, the scale as the dial.

- No pairs and no bank. The one pair is the prompt and its edit, which the in-place method already authors.
- The model computes what the edit means for this shot at this step, instead of receiving an average taken
  from other scenes.
- A scale above 1 exaggerates the edit, a scale between 0 and 1 gives part of it, and swapping the two
  prompts gives less of it.
- Cost: two DiT passes per step, the price the audio/video split already pays.

To find out first:

- Whether Pika tolerates it. Pika is a distilled recipe run without guidance, and distilled models often
  degrade when extrapolated. A small range of scales may be all it takes.
- Whether core's sampler carries the H3 conditioning (token tags, references, keyframes) on both prompts
  when their text lengths differ. Checked by reading after this was first written: it does (section 11.7).

A refinement from the literature (semantic guidance) keeps only the largest parts of the difference between
the two predictions, so an edit touches only the regions it is about. That would be new code, and only worth
it if the plain form works.

### 11.2 Prompt scheduling

Core's `ConditioningSetTimestepRange` (`<comfy>/nodes.py`) lets the original prompt drive the first steps
and the edited prompt the rest. That is the "lock the pose, then change the look" recipe the wiki describes
with a vector and a step window, done with two real prompts. It needs no new code, and section 11.7 found
nothing in core or in our nodes that blocks it.

### 11.3 Emphasis on words already in the prompt

The text rows sit in the same attention as the video and audio rows, so a bias on how strongly everything
attends to chosen tokens is a per-word dial: more of the fog, less of the neon sign. It needs a hook in the
attention path. Whether the pack's attention backends (`attention.py`, `sol_attn_h3.py`) can take one was not
checked.

### 11.4 Aim the vectors we already have

`control_vectors.py::meter_score` already scores which rows of a prompt respond to a vector. Applying the
shift only to the top-scoring rows, instead of every masked row, makes an existing vector less of a hammer
with no new data. It would be one more mask mode.

### 11.5 Where each fits

| Tool | Needs | Best at |
|---|---|---|
| Edit guidance | a prompt and an edit | precise changes to one shot, with a dial |
| Prompt scheduling | the same | keeping layout, changing look |
| Word emphasis | words already present | more or less of something written |
| Bank vectors | a concept, built once | a reusable look across many shots |
| Direction catalog (section 8) | nothing | finding controls nobody wrote |

If edit guidance holds up under Pika it could replace the single-prompt vector method, and the bank would be
kept for reusable looks only.

### 11.6 First test

11.1 and 11.2 need existing nodes only, so they can run before phase 3: one shot, one edited prompt, three
scales for 11.1 and one switch point for 11.2, each with a repeat, beside the plain prompt and the fully
rewritten prompt. It decides whether these two join the methods table in section 4.

The test graphs are hand-wired once through `workflows/build_workflows.py`, on the text-to-video and ref2va
graphs: the conditioning node twice (original and edited prompt), core's `CFGGuider` in place of
`BasicGuider` for 11.1, and `ConditioningSetTimestepRange` plus `ConditioningCombine` for 11.2. No control
vector node is in these graphs, so the test reads edit guidance alone. The image graphs wait on the batching
question in section 11.7.

### 11.7 What the code checks found

Two reads on 2026-11-09 against `main` at `856d4ac8`, with nothing run on the GPU: core's sampler, and every
custom node in the five control-vector graphs. No blocker was found for 11.1 or 11.2.

Core:

- Each conditioning gets its own H3 payload. `<comfy>/comfy/samplers.py::encode_model_conds` calls the
  model's `extra_conds` once per entry, and H3's `extra_conds` (`<comfy>/comfy/model_base.py`) builds the
  refined text, tags, references and a layout sized to that entry's own text length.
- Two different prompts run as separate passes. Entries are batched only when their text shapes match and
  their payloads are equal (`<comfy>/comfy/conds.py`, `CONDRegular.can_concat` and
  `CONDConstant.can_concat`).
- Each pass is labelled in `transformer_options["cond_or_uncond"]`.
- Timestep ranges are handled in `<comfy>/comfy/samplers.py::get_area_and_mult`, with nothing H3-specific.
- The text length sets where the video and audio rows start in position space, so two prompts of different
  length place the media rows at different offsets in their two passes. Each prediction is valid for the same
  latent, but part of their difference comes from the length change and not from the edit.

Our nodes:

- The conditioning nodes build a fresh conditioning per run, with its own tags, mask attribute and
  references (`conditioning.py`, `reference_conditioning.py`, `ph3_image_nodes.py`).
- `MiniMaxH3Sol` recomputes the sink boundary and its step gate on every call from that pass's layout
  (`sol_attn_h3.py`).
- `MiniMaxH3LoRABranch` and `MiniMaxH3Preflight` hold no state across passes (`lora_branch.py`,
  `preflight.py`).
- The control-vector wrapper takes its mask and length from each pass's own payload, and
  `control_vector_nodes.py::_step_in_window` has no counter that two passes could double-count.

Five things to design around:

1. A control vector steers both passes. The wrapper adds the vector to the original prompt's pass as well as
   the edited one (`control_vector_nodes.py::_make_control_vector_wrapper`; `cond_or_uncond` is read only
   for the live prompt text).
2. "Percent" has two meanings. The control-vector and LoRA windows count steps
   (`control_vector_nodes.py::_step_in_window`, `lora_branch.py`); `MiniMaxH3Sol` and core's timestep ranges
   go through `percent_to_sigma` (`<comfy>/comfy/model_sampling.py`), which depends on the sigma shift. The
   same number on both lands on different steps.
3. `MiniMaxH3Preflight` sees only the conditioning wired through it, so a second prompt's sequence length
   goes unreported unless it passes through Preflight too.
4. The reference graph pays twice: a second run of `MiniMaxH3ReferenceConditioning` repeats the picture
   encodes.
5. On the image graphs, two prompts of equal length might be batched together, and H3 refuses a batch above
   one (`<comfy>/comfy/ldm/minimax/model.py::_forward`). On the video graphs the payload carries a layout
   object that never compares equal, so the two are never batched. Whether the image graphs build that
   layout was not confirmed.

Also: with the audio/video split on, guidance means four DiT passes per step. The inline-words path that
resolves against the prompt subject gets an empty prompt on the original prompt's pass. Not determined:
whether the mask attribute reaches the wrapper at runtime (finding 13; a current render log without the
fallback warning confirms it), and the context-window sampling path, which rebuilds the payload.

### 11.8 The edit-guidance node

If the first test shows Pika tolerates the extrapolation, edit guidance goes behind one node of ours, so
that nobody wires two conditioning nodes into a core guider by hand. What it must do:

- Take one prompt and one edit (`old wording -> new wording`), and build the edited side itself with the
  same encoder, references and canvas. The two sides cannot differ in anything but the edit.
- Encode the references once and use them for both sides (point 4 above).
- Refuse when the old wording is absent or occurs more than once, or when the edit touches dialogue, tags
  or headers. The in-place compiler has most of these rules.
- Report the token count of both sides. If the model's layout allows the media rows the same starting
  position in both passes, do that; whether it does was not checked.
- Carry strength, a step window and a direction (more of, less of) as its only dials. A strength of 1 is
  the rewritten prompt exactly. A setting to use the original prompt until a given step covers 11.2.
- Count the window in steps, the convention the control-vector and LoRA windows use, and convert where core
  needs a noise level (point 2).
- Report both sequence lengths the way `MiniMaxH3Preflight` does for one (point 3).
- Make "control vector on the edited pass only, or on both" an explicit input (point 1).
- Keep the two prompts from being batched together on the image model (point 5).
- Write the edit, both token counts and the strength into the render's metadata.

A batch can then sweep strengths or edits over a shot by changing one string and one number.

## 12. Revised order: pair-free methods first

Added the same day, after the owner said that writing prompt pairs is not where they want to spend time, and
that it should be de-emphasized unless it is the only path that works. It is not the only path, but it is the
only one with render evidence today, so the order below tests the pair-free methods first and keeps the pair
files as the fallback. This section replaces the ordering in section 7 where the two differ.

1. The parts of phase 1 that inference needs: the tokenizer name, and a refusal when the scene mask is
   missing (findings 13 and 14).
2. The first test of section 11.6: edit guidance and prompt scheduling under Pika. This is the fork. The
   code checks in section 11.7 found nothing that blocks it.
3. If the fork goes well, the edit-guidance node of section 11.8.
   Then phase 4, the small apply node, with the aimed mask of section 11.4. Word emphasis (11.3) if the attention
   path can take a hook.
4. The direction track of section 8. Its cached rows can come from plain prompts in `prompt_bank/`; it does
   not need pairs.
5. Pairs last, and only as far as needed:
   - The existing files under `prompt_bank/control_vectors/` are rebuilt as they are, for reusable looks.
     No new pairs are written.
   - The rest of phase 1 (findings 1 to 4), phase 2 and phase 3 still apply to that rebuild, in the order
     section 7 gives.
   - The scene pool of section 5 and the prompt change of section 6 are parked. They come back only if a
     reusable look is wanted that no existing file covers and no pair-free method delivers.

What the fork in step 2 decides:

- If edit guidance holds up, the single-prompt vector method is dropped, and "author an edit" means typing
  `old wording -> new wording` once for the shot at hand.
- If it does not, the pair files remain the proven method, still without new authoring, and section 8 becomes
  the main route to new controls.

Under either outcome the methods table in section 4 loses "concept over a scene pool" as a default, and the
J-Lens words method stays as the instant, low-fidelity option.

## 13. Before executing: check the snapshot again

Much of the code this plan describes may have changed by the time it is executed. The findings cite
functions by name and not by line for that reason, but a name can move or go too. Start with
`git log --oneline 856d4ac8..HEAD -- control_vectors.py control_vector_nodes.py h3_pair_compiler.py h3_spans.py j_lens.py bench/control_vector bench/check_control_vectors.py workflows/build_workflows.py h3_rules.py`,
then go through these:

- Section 3: confirm each finding against the current code. A finding that no longer holds is struck, with
  the commit that fixed it, and the phase 1 item that answered it is dropped.
- Findings 1, 2 and 4 were shown by running the compiler on CPU; run them again and do not read them off
  this page.
- Section 9: derive the list of writers and readers again by grep. It is the part most likely to be stale.
- Section 11.7: the node audit covered the custom nodes in the five control-vector graphs as they were. Any
  of those nodes changed since gets its read repeated, the control-vector wrapper first.
- The graphs: confirm the five control-vector graphs still exist under those names, and which vector files
  they name.
- Section 12: the order holds whatever changed, since the fork in step 2 depends on Pika and on core, not on
  the control-vector code.

Where the code has moved, the code is right and this page is corrected or marked as overtaken.

## 14. Settings: the nodes, their defaults and what the graphs pass

Section 11.7 asked whether the custom nodes tolerate two prompts per step. It did not ask whether their
settings are right. That is a separate pass, done before the first test is queued and again when phase 4
changes the apply node.

The work uses the pack's own nodes throughout (`MiniMaxH3Conditioning`, `MiniMaxH3ReferenceConditioning`,
`MiniMaxH3AppendRefImage`, `MiniMaxH3Preflight`, `MiniMaxH3LoRABranch`, `MiniMaxH3Sol`,
`MiniMaxH3ApplyControlVectors`), never a hand-rolled substitute for one.

For each custom node in a graph the work touches, three things are compared, and any difference is either
explained or fixed in the generator:

- what the node's `define_schema` gives as the default;
- what the shared constant says (`h3_rules.py`, `workflows/h3_config.py`);
- what the generated graph passes, read from the graph as queued (`bench/what_h3_got.py`).

For `MiniMaxH3ApplyControlVectors` the pass reads every input, with what it does and whether its default is
right for each graph: `strength`, `mode`, `mask`, the step window, `allow_recipe_mismatch`, the audio and
video scales, the reference gate, and the inline words inputs.

Already seen in the graphs as generated at `856d4ac8`, to settle in that pass:

- The two video control-vector graphs render at `length` 362, which is `h3_rules.MAX_LENGTH`. That length
  is on the video grid but not whole on the audio clock: `h3_rules.on_both_clocks` is false for it. The
  arms in the results record were rendered at 141 frames, which is on both clocks. So the graphs ship at a
  length the evidence was not made at, and one the audio clock does not fit.
  Nothing is decided about it. [`../../masked_v2v/frames_rules.md`](../../masked_v2v/frames_rules.md),
  sections 0 to 2, has why the lengths on both clocks line up (the latent steps, the 17-frame VAE clips, the
  window); the longest of them is 345. Proposed for the first test of section 11.6: 141, the length the
  earlier arms used, so its clips can be compared with them. The generator's default length serves every use
  of the pack and is not changed by this plan.
- The graphs pass `strength` 1.15 where the node default is 1.0; the stacked graph passes 1.0 for each of
  two vectors, so its combined shift is larger than the single graph's.
- The ref2va and image-edit graphs pass `ref_vae_end_percent` and `ref_vae_aug` at their default values,
  which gates nothing.
- Every graph names a vector file that is not on disk.
- The image graphs pass no Pika LoRA and use the 4B encoder (finding 15).

The tracked checks that already cover part of this are run first, and their output is read, not just their
exit code: `bench/check_attention_defaults.py`, `bench/check_control_vectors.py`, and whichever checks
`docs/checks.md` lists for graph and node settings.

## 15. Corrections and additions after a second session read this plan

Added the same day. A second session read the plan against the code and the earlier research notes, and
disagreed with parts of it. Each point below was checked before it was accepted or kept.

### 15.1 Where this plan was wrong or incomplete

- The bare-frame comparison (E2.5) has a result. `exl3_qwen3vl/h3_control_vector_report.md` carries two
  bare twins beside their scene versions, with their norms and their cosine to each other in its tables.
  The first read of this plan looked in three files and missed it.
- The results record does have rewritten-prompt arms: `02_film16mm_edit`, `07_lettering_edit`,
  `31_ref2va_diner_explicit_pos` and `34_t2va_coffee_explicit_pos` in
  `bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`. It has no arm with a random vector of the
  same size; that part stands.
- `bench/control_vector/grading.py::_build_padding_prompt` is not stray prose (finding 17). It pads the base
  prompt to the edited prompt's token count so `grade` can separate the effect of the length change from the
  effect of the edit. Its filler sentence is a fixture and can live in a fixture file; the function stays.
- Finding 6 said holds are orthogonalized group means; that holds. The orthogonalization is the one in
  `control_vectors.py::hold_basis` as it stands when the plan is executed.

### 15.2 Changes accepted

- Multi-clause edits (finding 1, phase 1): pool the changed spans of one pair in a single pass, not one
  pair per clause. `h3_spans.py::differing_token_mask` already masks several separate spans; only the string
  check in `control_vectors.py::differing_span` refuses them. One pass also lets the edits see each other in
  the refiner. The check is relaxed only for the in-place path, where the clauses are declared; it keeps
  guarding pair files against accidental rewrites.
- The apply node (phase 4): keep one node on the model wire with the vector picked from a list, and add an
  optional vector socket for nodes that build or combine one. Inputs with no remaining use are dropped. This
  replaces the load, words and combine split as the default shape.
- The reference gate (finding 12, phase 4): no new node. The two inputs gate nothing at their defaults. The
  record's one arm with the gate turned down (`66_ref2va_diner_rubber_hose_refvae0p5`) reports no position
  jump and the subject's identity kept, with slightly more late stylization, so there is one arm for a use
  and none against. Whether the inputs stay as advanced inputs or go is decided when phase 4 is built.
- Edit guidance (sections 11.1, 11.6, 11.8): the first test also runs with the edit switched on after the
  first steps, not only across the whole schedule, so a clash between two slightly different layouts at the
  first steps is not mistaken for Pika refusing the extrapolation. `PackedLayout` in
  `<comfy>/comfy/ldm/minimax/model.py` starts the media rows at the text length, as section 11.7 says; the
  node should give both passes the same starting position. How it can do that without touching core is not
  yet checked.
- The image model and batching (section 11.7, point 5): settled by the second session's read. On an image
  graph with no references the two prompts' payloads compare equal, so two prompts of equal length are
  batched and meet the batch refusal. The node must make the two payloads differ.
- The bank mean (finding 8, phase 2): do not subtract it by default. The pair files are not a balanced
  sample of concepts, so their mean carries style as well as any length effect. A direction built from
  neutral pairs that differ only in how many words describe the same scene is the targeted test, and can be
  held out like any other hold.

### 15.3 Measured on the rebuilt vectors

The second session rebuilt the bank while this plan was being read, so the cross-vector cosines could be
taken from the vectors on disk, on CPU. Run when the rebuild has finished, from the repo root:

```
<comfy-venv-python> - <<'PY'
import glob, os, torch
from safetensors import safe_open
V, names = [], []
for f in sorted(glob.glob("../../models/control_vectors/*__fl2va__diff_span__lora_*.safetensors")):
    with safe_open(f, "pt") as s:
        v = s.get_tensor("vector.post_refiner").double()
    V.append(v / v.norm()); names.append(os.path.basename(f).split("__")[0])
V = torch.stack(V); G = V @ V.T
iu = torch.triu_indices(len(V), len(V), 1); off = G[iu[0], iu[1]]
print(len(V), "vectors; off-diagonal cosine mean, median, share positive:",
      float(off.mean()), float(off.median()), float((off > 0).double().mean()))
m = V.mean(0); m = m / m.norm()
print("cosine with the bank mean, min/median/max:", [float(x) for x in ((V @ m).min(), (V @ m).median(), (V @ m).max())])
print("chance scale for unrelated directions:", V.shape[1] ** -0.5)
PY
```

What it showed on the vectors present at the time (the rebuild was partway through): nearly every pair of
concepts has a positive cosine, far above the chance scale the last line prints, and every vector leans
toward the bank's mean direction. Across concepts, how strongly a vector leans that way rises with how much
longer its steered clause is than its plain one. So the shared component of finding 8 is there in the
post-refiner vectors the graphs load.

What this cannot tell: the cause. A longer steered clause and a stylized, high-intensity concept go together
in these files, so the lean may be length, style or both. The earlier table in
`exl3_qwen3vl/h3_control_vector_report.md`, section 2, was taken in encoder space on fewer vectors and shows
the same pattern of mostly positive entries with a few pairs near zero. The neutral-length direction of
section 15.2 is the test that separates the two causes.

### 15.4 Claims that did not hold as cited

Recorded so they are not carried forward:

- That the pooling modes were compared for collateral change. The record compares their norms and shows
  arms at matched push size (`39_t2va_coffee_head_swell_ffd_s3p5`, `40_ref2va_diner_zero_grav_ffd_s3p5`);
  no arm pair isolates collateral change by pooling mode. Pooling only the unchanged rows has not been built.
- That turning the reference gate down was shown to corrupt the first frames. The arm cited above reads the
  other way.
- That adding a bias to attention for a word equals `amplify` on that word's rows. `amplify` in
  `control_vectors.py::apply_post_refiner` scales only the part of a row along the vector. It may serve as a
  rough stand-in for section 11.3 and would be labelled as one.

### 15.5 The system prompt after the second session's commit

`docs/wiki/system_prompts/h3_control_vector_system_prompt.md` at `7c01c792`:

- It introduces `completely_preserved` as a retention degree and uses it in the ref2va example.
  `grep -c completely_preserved vendor_guides/ref_en.md` prints zero; the guide's term is `fully_preserved`.
  A model copies the example, so this teaches a term H3 was not trained on. To be reverted.
- Its in-place mode asks for clauses separated by semicolons, which the compiler still refuses for two
  description clauses and splits wrongly inside prose (findings 1 and 4). It comes right when 15.2's first
  item lands.
- `bench/control_vector/llm_pairs.py` carries three different defaults for the output token limit (the CLI
  flag, `_generate_pairs_with_llm`, and the fallback in `cmd_generate_pairs`). Ten pairs from the existing
  files fit the CLI default with little room at the long end; one constant should hold it.
- The task itself is unchanged: every scene written twice, identical but for one clause. Section 6 stays
  parked under section 12.

## 16. No prompts or prompt fragments in code

The owner's rule: prompt text does not live in code. The pack's generator already works that way: every
prompt a graph ships with is a `prompt_bank/` entry loaded by id (`workflows/build_workflows.py::_bank_prompt`).
The control-vector code does not yet.

### 16.1 Where prompt text sits in control-vector code today

Found by walking each file's string literals for H3 prompt markers, leaving out docstrings and help text.
Counts move as the code moves; the walk is the observable, and 16.3 makes it a tracked check.

- `bench/check_control_vectors.py`: the largest holder. Whole example prompts are written inline in three
  checks: `check_spans_partition_and_refusals`, `check_pair_file_loader_and_refusals` and
  `check_runtime_nodes_arm_and_cli`. They are there because each check needs a prompt to feed the tokenizer,
  the loader or the compiler, and one was typed in each time. They are test inputs, not model behavior, but
  they are hand-written prose in the vendor format that nothing keeps current.
- The canned scene list in `control_vector_nodes.py` (removed in `3f35901e`): three canned ref2va scenes that ran at build time for the
  image method, each asserting a person in the picture.
- `bench/control_vector/grading.py::_build_padding_prompt`: one filler sentence used to pad a prompt to a
  length.
- `j_lens.py`: word lists that guess intent from a prompt (`AGAINST_PRESETS`, `_REPLACEMENT_CONCEPT_WORDS`,
  `_SHORT_WORD_FALLBACKS`, `_SUBJECT_STOPWORDS`, `_PROMPT_RECEPTOR_STOPWORDS`, `_MEDIUM_KEYWORDS`). These are
  words, not prompts, and the walk for markers does not see them; they are the same habit in another form.

Not in scope here, and left alone: `masked_prompt_text.py` assembles the masked lane's prompt from clauses
by design, under that lane's own rules, and short format markers in validators (a section header name, a
tag pattern) are the format, not a prompt.

### 16.2 What replaces each

- Checks take their prompts from `prompt_bank/` by id, the way the generator does, or from fixture files in
  a folder beside the check. A refusal case is made by altering a bank prompt in code (remove a section,
  repeat a clause, change whitespace), so the case says what it breaks and no second prompt is typed.
- The image method takes the caller's own prompt and refuses without one; the canned scenes go.
- The padding sentence becomes a fixture file the function reads.
- The J-Lens word lists leave the sampling step. Any that remain for the readout become data files with
  their provenance beside them.

### 16.3 A check so it does not come back

A tracked `bench/check_no_prompt_literals.py`: the same walk, over the control-vector code files, red on any
string literal that is prompt-shaped (an H3 section header or tag inside a literal long enough to be a
sentence), with a short allowlist that gives a reason per entry. This is a hard failure of the kind the
repo allows a red for: a literal is there or it is not. It is added when 16.2 is done for a file, one file
at a time, so it is never born red.

## 17. J-Lens: the bundle, its builder and the research folder

Read on 2026-11-09 against `7c01c792`. Nothing was run on the GPU. The bundle on disk was not rewritten by
that commit; its file time is the observable.

### 17.1 The builder is half a builder (corrects 0.2 and finding 10)

`bench/control_vector/j_lens_builder.py::cmd_build_j_lens` recomputes the refiner half of the bundle
(`Sigma_49_p030`, `J_ref_*`, `J_wiener_*`). The lexical half (`J_49_to_63`, `M_union`, `cloze_base_union`,
the index tensors and the word list) is copied out of the bundle that is already on disk, and the command
refuses without one. Its docstring names a `--qwen-64l-model` option for computing that half; no such
argument exists in the builder or in `bench/control_vector/cli.py`. So the lexical half still cannot be
made again from tracked code, and finding 10 is open for it.

Other points in the builder:

- By default it writes over the bundle it reads the lexical half from. The only copy of the part that
  cannot be rebuilt is the file being replaced.
- If the Pika LoRA spec fails to parse, a bare `except` falls back to no LoRA and the build goes on. With
  Pika as the default recipe that is a silent change of recipe.
- The file's metadata carries a format tag, a description and the word list. It does not say which
  checkpoints, LoRAs, prompts or constants made it.
- Its sample is the first prompts of `prompt_bank/bank.json` in file order, and the first tokens of each
  (`--num-prompts`, `--max-tokens`). The research note describes a stratified sample. So a rebuild would not
  reproduce the bundle on disk, and the covariance is estimated from the openings of prompts, with fewer
  rows than it has dimensions.

### 17.2 What to change

- Write the lexical half's builder, from the full 64-layer weights, or state plainly that it is not
  reproducible and keep a second copy of the bundle. Until then the builder writes to a new file and never
  over its own source.
- A failed LoRA spec is a refusal.
- The bundle records its recipe, the ids of its sample prompts, its constants and its build date.
- The sample becomes whole prompts, more of them, spread over the three formats. This is also the row
  sample the direction track needs (section 8, D1), so it is made once and shared.
- The refiner half is rebuilt after the mask and the apply node settle (0.1, step 3). Its Jacobian assumes
  how much of the attention falls on steered rows, and the narrower `scene_only` mask of `856d4ac8` changes
  that, most on ref2va prompts.
- In `j_lens.py`: the word lists leave the sampling step (section 16); `_DEFAULT_RECIPES` repeats file
  names that `workflows/h3_config.py` holds and should read them from there; `refiner_singular_vectors`
  recomputes a full decomposition on every call and should keep it; the fitted constants (the covariance
  exponent, the two negative weights) carry where they came from beside them.
- J-Lens is not needed for the fork of 0.1, step 2. None of this blocks it.

### 17.3 The research folder `exl3_qwen3vl/`

It holds three kinds of thing:

- Recipes and a calibration builder for converting the full Qwen3-VL-32B to ExLlamaV3 (`*.yaml`,
  `build_h3_cal_data.py`, `runbook.md`). Nothing changed today affects these. The calibration file and the
  converted models do not need making again.
- A labeller and a viewer builder (`label_h3_control_vectors.py`, `build_control_vector_viewer.py`). These
  are tools kept under `docs/`. `j_lens.py` and the builder lean on the labeller's word map. If the lexical
  half's builder is written, the word map moves into tracked code beside it.
- Outputs from the vectors of 2026-11-05: `h3_control_vector_labels.json`, `h3_control_vector_report.md`,
  `control_vector_viewer.html`. They describe fourteen vectors in encoder space under the older pooling,
  two of them the retired bare twins. They are a dated record and stay as one. They are not a description
  of the rebuilt bank, and pages that link them as current should say so. Labels for the rebuilt bank would
  be a new run, and the CPU readout in `j_lens.py` now covers the lexical part of it.

### 17.4 Deciding the lexical half by one comparison

The owner's word on this: do it the right way, and do not keep an experiment for its own sake.

Rebuilding the lexical half means loading the full 64-layer model and writing again a method no tracked code
holds, to get a table that still points at "about to say this word" (finding 9). The simpler table is built
from what H3 sees: for each word, the mean of its rows across real prompts, less the mean of all rows. It
needs only the shipped 50-layer encoder and the refiners.

Both tables label the rebuilt vectors, and the one that names the concepts better stays:

- If the simple table holds up, the lexical half, the Jacobian to the output layers, the cloze baselines,
  the fitted exponent and negative weights, and the word map all go, and `exl3_qwen3vl/` goes with them
  except `h3_control_vector_report.md`, which holds results this plan cites.
- If it does not, the lexical half gets a real builder from the full weights in the model store, the word
  map moves into tracked code beside it, and `exl3_qwen3vl/` still goes, apart from the report.

Either way the viewer, its builder and the labels file go: they show fourteen vectors that no longer exist
in that form.

The rows come from `bench/control_vector_rows.py`: one encode pass over both sides of every pair file and
the text-only bank prompts, caching each token's encoder row, both families' post-refiner rows, its token
id, its vision tag and whether `scene_only` covers it, under the untracked `data/`. The same cache feeds the
pooling comparison and held-out score of phase 2 and the candidates of section 8, so those need no second
pass.

## 18. The wiki pages: what is rewritten, and by what rules

The pages under `docs/wiki/control_vectors.md`, `docs/wiki/control_vectors/` and
`docs/wiki/system_prompts/h3_control_vector_system_prompt.md` were rewritten in `7c01c792`, before the fixes,
the fork and the CLI change. Most of what they say about commands, defaults and methods will be wrong again
when those land, so they are rewritten once more, after step 6 of 0.1 and not before.

### 18.1 What is corrected now, with step 1

- `completely_preserved` in the system prompt and the examples page (0.4).
- The hold description that names `torch.linalg.qr` (0.4).

### 18.2 Rules for the rewrite

- A capability is listed only with a link to the arm that showed it. `grep -c "bench/results"` over the
  current pages shows how few such links there are. What has no arm is listed as untested, in its own table.
- No figure in prose. A strength, a norm, a timing or a count points at the constant or the command that
  holds it. The recommended strengths in the current pages are examples of what goes.
- Code is cited by file and function, never by line.
- Reference tables are generated, not written. The command and flag tables come from the parser, and the
  node input tables from each node's `define_schema`, by a small tracked tool with a `--check`, the way
  `bench/build_index.py` keeps the indexes. A hand-written flag table is a second copy that drifts.
- Each method's page states what it estimates, what it is good for and its main limit, from section 4 as
  corrected by sections 15 and 17.
- The rewrite is done by a reader who did not build the code, and `docs/wiki/decisions.md` records what the
  pages used to claim.

### 18.3 The pages

| Page today | What happens to it |
|---|---|
| `control_vectors.md` (the hub) | Rewritten: what steering is here, the methods in one table with their evidence, and where to start. Its tables of vector types and recommended settings go unless each row has an arm. |
| `control_vectors/building_pairs_and_vectors.md` | Rewritten around the one build path: where pairs come from, pooling as chosen by phase 2, holds as the code builds them, the shared component and what was done about it. |
| `control_vectors/preset_catalog_and_cli_reference.md` | The hand-written command table is replaced by the generated one. The list of pair files stays as a list, with no claims per file. |
| `control_vectors/comfyui_nodes_and_controls.md` | Rewritten for the apply node's final shape, its input table generated. |
| `control_vectors/examples_and_recipes.md` | Kept only as worked examples that each link an arm. The rest goes. |
| `control_vectors/j_lens_diagnostics_and_word_lab.md` | Rewritten after section 17.4 decides what J-Lens is. |
| `control_vectors/llm_prompting_guide.md` and `system_prompts/h3_control_vector_system_prompt.md` | The prompt becomes a data file the page links to (section 6). The guide shrinks to how to run the generator, or goes if pair generation stays parked. |
| `control_vectors/studio_ui_guide.md` | Not rewritten. One line at its top says the UI is not maintained at present. |
| a new page | Steering without a vector (section 11), if the fork of 0.1 step 2 goes well: edit guidance and prompt scheduling, with the node of section 11.8. |

The research notes under `docs/research/control_vectors/` are dated records and are not rewritten. Where a
wiki page presents one of them as current, the page is what changes.

## 19. The CLI: commands, flags and the code behind them

### 19.1 Where it is

`7c01c792` named the creation methods as commands, which was asked for, but by adding to what was there.
`bench/control_vector/cli.py::build_parser` is the observable: three commands (`from-prompt`, `build` with
its alias `from-toml`, and `make`) each build a vector from words on a target prompt; `build` also takes
the flags of the words method; `from-md`, `generate-pairs`, `rebuild` and the aliases sit beside them. Many
flags are second names for one thing (`--name` and `--output-name`, `--out` and `--out-dir`, `--dit` and
`--unet`) or belong to code that is going (`--target-weight` for the Pair 0 splice, `--frame`, the test
prompt output).

### 19.2 The target

One command per thing a person does, each with only its own flags, and one shared group for the recipe:

| Command | Does | Replaces |
|---|---|---|
| `from-prompt` | builds from an in-place edit of one prompt | the words flags on `build` and `make`; dropped altogether if edit guidance takes this method's place |
| `from-pairs` | builds from a pair file, or every pair file | `build`, `from-toml`, `make`, `rebuild`, `from-md` |
| `from-words` | builds from single words | `words` |
| `from-images` | builds from image pairs and the caller's prompt | `make-media` |
| `combine` | adds, inverts or holds out existing vectors | `sculpt` |
| `inspect`, `grade` | read a vector; measure it against the edit and the nulls | themselves |
| `rows` | the encode pass of section 17.4 | `bench/control_vector_rows.py` as a command |
| `lens` | builds or verifies the J-Lens data, in whatever form 17.4 leaves it | `build-j-lens` |
| `pairs` | has an LLM write the edits for a concept | `generate-pairs`; present but marked parked |

- The recipe group is declared once and shared: families, checkpoint, encoder, LoRA, devices, output
  folder, name, overwrite. No second names. The old command names are removed, not aliased; this is a
  tinkering repo and the graphs do not call the CLI.
- Each command's help says, in a few lines, what the method estimates, what it is good for and its limit.
  The same text sits in a comment beside the function that does the work, and the wiki's method pages are
  built from it (18.2), so there is one wording.
- A flag is kept only if something reads it and a person would set it. The leftovers lens applies: what
  does each cover, does that still hold, what retires it.

### 19.3 The code behind it

- `bench/control_vector/build_cmd.py` and `bench/control_vector/make_cmds.py` become one build function that
  takes pairs and a recipe, with each command a thin caller. Today both files hold a copy of the encode,
  pool and save loop.
- The silent fallbacks in that loop become refusals (0.4).
- `bench/check_control_vectors.py` loses the cases for commands that no longer exist and takes its prompts
  from files (section 16).
- Writers and readers of anything that changes are listed again by grep before the change lands (section 9
  is the pattern).

### 19.4 The UI

The Studio UI under `bench/control_vector/` (`ui_server.py`, `ui_state.py`, `web/`) is left for another day.
It imports functions from the files 19.3 merges. It is not kept working through this, and no dead function
is kept alive for it: where an import it needs is gone, the `ui` command refuses with one line saying the UI
is not maintained at present. `studio_ui_guide.md` says the same.

## 20. Outcomes of the first measurements

Added the same day, from the row cache of section 17.4 (every pair file, both sides, and the text-only bank
prompts). Each figure is printed by the tool named; none is repeated here.

### 20.1 The two label tables: both stay

`bench/control_vector_lexicon.py` printed both labels for every rebuilt vector, with each concept's own pair
file left out of the rows table.

- J-Lens names the concept in whole words for most of them, including words that never occur in our
  prompts. It is vague on a few.
- The rows table is rougher: it returns token pieces, knows only words our prompts contain, and borrows
  wording from neighboring concepts. It names a few concepts that J-Lens misses.
- The rows table's minus side reads much the same words across about twenty concepts (holds, maintains,
  standard, normal, neutral, natural). That is the shared plain side of section 0.3, seen directly.

So section 17.4 resolves to its second branch. The lexical half is worth keeping and gets a real builder
from the full weights in the model store. It depends only on the base text encoder, not on the recipe, the
refiner or the mask, so it is built once, and the bundle on disk is there to check the rebuilt tensors
against. The rows table stays as a second readout and is the labeller section 8 (D3) asks for. The
`exl3_qwen3vl/` folder goes as section 17.4 says, the word map moving into tracked code with the builder.

### 20.2 Pooling: the default stays

`bench/control_vector_pooling.py` pooled every pair four ways and printed, per way, the held-out score, the
same score with the other concepts' mean direction projected out, the mean cosine between concepts, and the
cosine with the vector on disk.

- The cache reproduces the built vectors, so the figures describe the vectors the graphs load.
- The three existing ways (`diff_span`, `from_first_diff`, `scene_text`) find nearly the same direction and
  differ mainly in length, which the apply node normalizes. Nothing here argues for changing the default.
- Pairs within a concept agree with each other, and still agree once the shared part is projected out. Each
  vector is a concept direction with a shared component on top.
- The shared component is about the same under every way of pooling. It is a property of the pairs, not of
  the pooling.
- Pooling only the unchanged rows gives a different direction, much smaller and consistent from pair to
  pair.

Phase 2 is answered: `diff_span` stays, and no bank mean is subtracted.

### 20.3 The apply node, option by option

A one-off test drove the wrapper with synthetic rows through every mode, mask, window and stacking path,
against the phase 1 code (`3f35901e`). Everything behaved as its help text says except the two rows added
to 0.4. Read but not run: one `strength` serves both the file vector and the inline words when both are
set; a vector is applied on a pass marked as the negative prompt too; the reference gate takes the lowest
value across stacked nodes; the `strength` range reaches far beyond a row's own length. These go into the
apply node's reshape (section 15.2).

### 20.4 Changes to earlier sections

- Section 19.2: `from-prompt` stays whether or not edit guidance works. It costs one model pass a step where
  guidance costs two, and it works with masks and stacking. Rebuilding a vector from the pairs stored in its
  own metadata, with no pair file, also stays.
- Section 11.8: padding the shorter prompt is not a way to give both passes one position origin. Core's
  attention takes no mask (`<comfy>/comfy/ldm/minimax/model.py`, where `optimized_attention` is called with
  `mask=None`), and tag 0 marks vision rows, so padded rows would be attended. Edits of equal token count
  work today; shifting the layout's positions in the wrapper is unchecked.
- Section 11.6: the test gains arms whose edit has the same token count as the text it replaces, so that a
  failure cannot be put down to the length change. The late-start arm does not remove the length difference
  on the steps where guidance is on; it only keeps the first steps on one prompt.
- The post-refiner row length the plan relies on is the one the cache gives, the same as the constant in
  `j_lens.py` (`DEFAULT_POST_REFINER_MEDIAN_ROW_NORM`).

## 21. The fork passed: building the edit-guidance node

The owner looked at the stacked clips of both test shots on 2026-11-09 and passed the fork
(`docs/wiki/decisions.md`, that date). Step 3 of 0.1 is now the next build.

### 21.1 What the test showed, and what it did not

Seen by the owner on playback: both shots hold together in every arm; on the diner shot the two guided arms
are brighter than the rest, which the edit asked for. Seen in single frames by a session, not a judgement:
the guided arms carry the edit further as the scale rises, with more saturated color at the higher scale on
both shots; the scheduled and late-start arms keep the base shot's framing and take on less of the edit.
A mean pixel difference from the base and from the rewrite was taken per arm on sampled frames; it says how
far a picture moved and nothing about how it looks.

Not judged yet: sound in any arm, and the same-length control arms (their stacks are beside the others).

### 21.2 Shape of the build

Two nodes, both small, and no change to core:

- A prompt edit node. It takes a prompt and edit lines (`old wording -> new wording`, one a line) and gives
  the edited prompt and a short report with both token counts. It reuses the in-place compiler's parsing and
  its refusals (wording absent, wording that occurs more than once, an edit inside dialogue, a tag or a
  header), so there is one copy of those rules.
- An edit guider node. It takes the model, the original conditioning and the edited conditioning, a scale
  and a first guided step, and gives a guider for `SamplerCustomAdvanced`. It is a subclass of core's
  `comfy.samplers.CFGGuider` with its own `predict_noise`, the way core's own `Guider_DualCFG` is made
  (`<comfy>/comfy_extras/nodes_custom_sampler.py`).

For ref2va the references are encoded once by `MiniMaxH3EncodeReferences` and each prompt is laid on them by
`MiniMaxH3PromptOnReferences` (`reference_encode.py`), which answers the double encode of section 11.7.

What the guider does that the hand-wired test could not:

- Before the first guided step it runs the original prompt alone: one model pass, not two with a zero
  difference.
- It counts in steps, read from the schedule in `transformer_options`, the convention the control-vector and
  LoRA windows use. No conversion to a noise-level percent.
- It gives both passes one position origin by shifting the media rows of the shorter prompt's layout. The
  second session read that `PackedLayout.position_ids` is a plain tensor, that `_forward` compares only the
  layout's signature, and that the rotation frequencies are read from those positions
  (`<comfy>/comfy/ldm/minimax/model.py`). To be confirmed by a test that compares the positions the two
  passes actually use.
- It makes the two payloads differ, so two prompts of equal length are never batched on the image model.
- It refuses when either conditioning lacks the H3 token tags, or when the two differ in anything but text
  (canvas, length, references).
- It writes the scale, the first guided step and both token counts where the render's metadata keeps them.

### 21.3 Open choices, settled when the node is built

- Whether a control vector applies to the edited pass only or to both. Today the wrapper steers every pass
  (section 20.3). The guider can mark its passes so the wrapper can tell.
- The scale's range. The test ran 1.5 and 2.5; the higher one adds color beyond the edit. A default near
  the low end and a range that stops short of the slider extremes the apply node has.
- What the generator ships: one text-to-video and one ref2va graph using the two nodes, in place of
  hand-wiring. `bench/edit_guidance_arms.py` then builds its arms from those graphs.

### 21.4 Both ends

New: the two node classes, their registration, `bench/node_id_manifest.json`, the generator entries, a check
for the guider's arithmetic on synthetic predictions (a scale of 0 gives the original pass, 1 the edited
pass, and before the first guided step only one pass is made). Read by: `SamplerCustomAdvanced` (core),
`MiniMaxH3Preflight` if the edited conditioning is passed through one, and the control-vector wrapper, which
sees two passes a step once the guider is on.

## 22. What has been fixed since this plan was first written

Kept here so sections 3 and 0.4 can stay as they were found. All on `main`.

| Finding or defect | Fixed in | How |
|---|---|---|
| 1, several clauses in one in-place edit | `3f35901e`, `229dfae3` | the changed spans are pooled in one pass; pair files keep the single-clause rule unless their `[vector]` table sets `allow_multi_span` |
| 2, the image frame the validator rejected | `3f35901e` | image prompts compile with the scene frame |
| 3, the Pair 0 splice and its truncation | `3f35901e` | the helper is removed; a target prompt with a pair file and no words is refused |
| 4, a pole that occurs twice, `;` inside prose | `3f35901e` | both refused or handled; a pole must occur exactly once |
| 7, a consistency of 1.0 for one pair; the canned image scenes | `3f35901e`, `229dfae3` | one pair reports no score; the image method requires the caller's prompt |
| 13, the mask attached inside a bare `except`, falling back in silence | `3f35901e` | the mask travels as a `model_conds` entry and a missing mask is refused; the two older homes for it remain until a render with them removed confirms the route (a full sampler loop on CPU with both deleted has confirmed it) |
| 14, the tokenizer looked up under one name | `3f35901e` | one helper resolves the 32B and 4B tokenizers |
| 15, the image graphs on the 4B encoder | `842a4108` | the image graphs default to the 32B encoder |
| the silent no-LoRA fallback in the build | `3f35901e` | a LoRA that does not resolve is a refusal |
| the three token-limit defaults | `3f35901e` | one constant |
| `completely_preserved`, and the hold description | `3f35901e` | corrected in the pages |
| a hold node with a different mask; `hold_words` alone | `229dfae3` | both refused |
| the row cache tool reading the mask off a tensor attribute | `797ccd11` | built from the token ids |

Still open from those lists: findings 5, 6, 8 to 12 and 16 to 18 as the later sections treat them, the same
fallback in `bench/control_vector/j_lens_builder.py`, a hold node with no add node at all (it does nothing,
with no message), and the compiler's test for several spans, which looks for the joining mark in the span
text.
