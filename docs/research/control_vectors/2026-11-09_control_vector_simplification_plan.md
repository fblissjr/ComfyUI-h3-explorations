# 2026-11-09: control vectors, a plan to simplify, measure and extend

A dated record. It proposes; nothing in it is built. It was written from a read of `main` at `856d4ac8`
(code, the pair files, the wiki pages and one results record) plus CPU-only runs of the pair compiler.
No GPU build and no render was run for it, and each finding below says how it was checked.

Related: [`2026-11-06_j_lens_and_refiner_jacobian.md`](2026-11-06_j_lens_and_refiner_jacobian.md),
[`../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md),
[`../../wiki/control_vectors.md`](../../wiki/control_vectors.md).

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
| 3 | A pair file given with `--target-prompt` gets a spliced "Pair 0", returns unchanged in silence when the clause is absent, and keeps only `pairs[:9]` | `bench/control_vector/helpers.py::_inject_target_prompt_into_pairset` | read |
| 4 | A pole that occurs twice is replaced at its first occurrence only; a `;` inside prose splits the clause; with several scenes the substring must be in every one | `h3_pair_compiler.py::_apply_clause_inplace`, `_split_clause_statements` | run |
| 5 | Every text method pools only the rows whose wording changed, then adds one vector to every masked row | `h3_spans.py::pooling_mask`, `control_vectors.py::apply_post_refiner` | read |
| 6 | Holds are one mean difference per group, orthogonalized; the wiki calls it an SVD basis | `control_vectors.py::hold_basis` | read |
| 7 | One pair reports a consistency of 1.0; the image node repeats one image pair over canned scenes so the score is defined | `control_vectors.py::quality_report`, `control_vector_nodes.py::DEFAULT_MEDIA_SCENES` | read |
| 8 | In nearly every pair file the steered clause is longer than the plain one, so vectors may share a component | load each file with `control_vectors.load_pairs` and compare `a_span` and `b_span` lengths | run; the shared component itself is not measured |
| 9 | J-Lens word directions are the output head times a Jacobian, the direction that makes the encoder predict a word next; its exponent and negative weights were fitted against the bank | `j_lens.py::synthesize_from_words`, section 6.1 of the J-Lens note | read |
| 10 | No tracked file writes the J-Lens bundle | `git grep -l J_49_to_63` returns readers and docs only | run |
| 11 | The apply hook is placed correctly: core refines text once per sampling, so the wrapper sees post-refiner rows and only edits `context` | `<comfy>/comfy/model_base.py` (`extra_conds`), `control_vector_nodes.py::_make_control_vector_wrapper` | read |
| 12 | The apply node does five jobs: load a vector, synthesize from words, hold words, an audio/video split that runs the DiT twice per step, and gating reference latents | `h3_rules.py::CONTROL_VECTOR_DEFAULTS`, `CONTROL_VECTOR_OPTIONAL_DEFAULTS` | read |
| 13 | The scene mask rides on a Python attribute of the tags tensor, set inside `except Exception: continue`; when it is absent `scene_only` falls back to all text rows with a log line | `h3_spans.py::attach_span_masks`, `control_vector_nodes.py::_resolve_custom_inference_mask` | read |
| 14 | The span code looks the tokenizer up as `qwen3vl_32b`; the image encoder registers as `qwen3vl_4b`, so the image graphs get no mask | `h3_spans.py::attach_span_masks`, `<comfy>/comfy/text_encoders/minimax.py::PH3ImageTokenizer` | read |
| 15 | The image graphs apply a vector built from 32B rows under the Pika recipe to rows from the 4B encoder with no Pika LoRA | `workflows/image/h3_p_h3_image_*_control_vector_api.json` | read |
| 16 | Graphs name a vector by pooling and LoRA hash, and a resolver swaps siblings by filename | `workflows/build_workflows.py` (the `control_vector=` entries), `control_vector_nodes.py::resolve_vector_file_path` | read |
| 17 | Hardcoded prose remains: three canned ref2va scenes that assert a person, the J-Lens word tables, a filler sentence in grading | `control_vector_nodes.py::DEFAULT_MEDIA_SCENES`, `j_lens.py::AGAINST_PRESETS` and the stopword tables, `bench/control_vector/grading.py::_build_padding_prompt` | read |
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
- Two control arms in that round, which the results record does not have: a random vector of the same size
  as the steered one (how much does any large shift change the shot?), and the prompt simply rewritten (the
  thing every vector competes with).
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
  [`2026-11-04_h3_control_vectors.md`](2026-11-04_h3_control_vectors.md): no result was found.
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
  when their text lengths differ. Not checked.

A refinement from the literature (semantic guidance) keeps only the largest parts of the difference between
the two predictions, so an edit touches only the regions it is about. That would be new code, and only worth
it if the plain form works.

### 11.2 Prompt scheduling

Core's `ConditioningSetTimestepRange` (`<comfy>/nodes.py`) lets the original prompt drive the first steps
and the edited prompt the rest. That is the "lock the pose, then change the look" recipe the wiki describes
with a vector and a step window, done with two real prompts. It needs no new code. The same question about
H3 conditioning on both prompts applies.

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

## 12. Revised order: pair-free methods first

Added the same day, after the owner said that writing prompt pairs is not where they want to spend time, and
that it should be de-emphasized unless it is the only path that works. It is not the only path, but it is the
only one with render evidence today, so the order below tests the pair-free methods first and keeps the pair
files as the fallback. This section replaces the ordering in section 7 where the two differ.

1. The parts of phase 1 that inference needs: the tokenizer name, and a refusal when the scene mask is
   missing (findings 13 and 14).
2. The first test of section 11.6: edit guidance and prompt scheduling under Pika. This is the fork.
3. Phase 4, the small apply node, with the aimed mask of section 11.4. Word emphasis (11.3) if the attention
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
