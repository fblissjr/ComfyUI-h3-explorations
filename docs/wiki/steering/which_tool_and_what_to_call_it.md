# Which Tool Is Which, and What to Call the Word Readout

Last updated: 2026-11-21.

This pack has four parts that get mentioned together: **Prompt Expander**, **Control Vectors**, **Edit Guidance**, and the **J-Lens**. Prompt Expander authors complete `1x` prompts; Control Vectors and Edit Guidance steer renders; J-Lens mostly reads.

Read this page first if you are explaining the pack to someone else. For how `MiniMaxH3Steer` erases prompt conflicts in `1x` (`add_norm_matched_erasure`) see [`how_pole_a_erasure_works.md`](how_pole_a_erasure_works.md). For the step-by-step mechanism of the word readout see [`how_the_word_readout_works.md`](how_the_word_readout_works.md). *(Early Q&A examples archived in [`../archive/2026-11-12_control_vector_worked_examples.md`](../archive/2026-11-12_control_vector_worked_examples.md).)*

---

## 1. The four, side by side

| | Video Prompt Expander (`MiniMaxH3PromptExpander`) | Control vector (`MiniMaxH3Steer`) | Edit guidance (`MiniMaxH3EditGuider`) | Word readout (J-Lens) |
|---|---|---|---|---|
| **What it is** | Local `heylook` authoring (`@extend:prepend\|append` overlays, trailing `ref_end_percent`) + Python grammar contract ([`h3_prompt_contract.py`](../../../h3_prompt_contract.py)) | One steering direction + rank-8 Pole-A subspace eraser (`add_norm_matched_erasure`) applied to selected text tokens | Two prompts compared at every guided step | A translator between a vector and English words |
| **Where it acts** | Before `Qwen3-VL` conditioning (writes the prompt itself) | After the token refiner, before the DiT (`token_focus = 1.0`) | Inside the DiT's predicted update | Nowhere in the render, unless wired as `MiniMaxH3WordVector` or `hold_words` |
| **Built from** | Short scene idea + concept card + wired media/length | 10 paired contrast prompts (`--families fl2va`) storing `vector.post_refiner`, `erasure.mu_d`, and rank-8 `erasure.vr` (`~193.5 KB`) | Your `original` prompt and edits (`MiniMaxH3PromptEdit` or `MiniMaxH3AutoEditPrompt`) | The text encoder's weights, once, into a bundle file |
| **Render cost** | One `Qwen3-VL` + one `1x` `BasicGuider` pass | One model pass (`1x`), the same as no steering | A second model pass on every guided step (`2x`) | None. It runs on the CPU |
| **Good for** | Fresh multi-shot scenes, mutually exclusive lighting looks, wardrobe, blocking, dialogue | Reusable orthogonal film-stock, medium, material, and acoustic-timbre dials (`bank/`) — even when an un-rewritten prompt mentions natural color or daylight | Keeping an existing take/seed while tweaking weather, `audio_only`, or a `[F, H, W]` mask | Checking a vector by eye before rendering, scrubbing a side effect, sketching a vector from single/multi-token words (`Model2Vec`) |
| **Cannot do** | Preserve step-0 latent trajectory of an unrelated existing render | Re-choreograph limbs, swap multi-sentence dialogue scripts, override Dual-Lock `fl2va` (`cond_video_latents` + `tags == 0` vision chroma lock) / unlock bright `<Picture 1>` garments while `ref_end_percent = 1.0` pins color, or preserve full monochrome when 3+ heavy visual sliders compete on `scene_only` | Be saved and reused across arbitrary scenes | Replace 8–12 pair contextual `.toml` erasure on hard global palette/material overrides (Ablation 6 `< 2 ms` sketchpad boundary) |

Where each lives in the code:

- **Video Prompt Expander (`1x` Easy Button)**: [`h3_prompt_expander.py::MiniMaxH3PromptExpander`](../../../h3_prompt_expander.py) with [`h3_prompt_contract.py`](../../../h3_prompt_contract.py). Design note: [`heylook_auto_edit_guidance_design.md`](heylook_auto_edit_guidance_design.md).
- **Control vector**: `steering/nodes.py::MiniMaxH3Steer` applies one (supporting native `fl2va`, `ref2va`, `h3_image`, and `fl2va_base` `P-H3-Image`). The hub page, [`../steering.md`](../steering.md), says how they are built and which rows they touch.
- **Edit guidance**: `edit_guidance.py::MiniMaxH3EditGuider` with `edit_guidance.py::MiniMaxH3PromptEdit` or [`auto_edit_prompt.py::MiniMaxH3AutoEditPrompt`](../../../auto_edit_prompt.py). [`../edit_guidance.md`](../edit_guidance.md) has the dials and the pass counts.
- **Word readout**: `steering/xray.py::decode_vector` reads, `steering/xray.py::apply_soft_holds` scrubs, and `steering/xray.py::synthesize_from_words` builds. The node that builds is `steering/nodes.py::MiniMaxH3WordVector`.

### How they relate

- A vector made by the word readout **is** a control vector. It goes through the same Steer node. The readout
  is a second way to make one, from a word list in place of prompt pairs.
- Edit guidance uses no vector and no readout. It is a different mechanism with a different cost.
- Neither a control vector nor edit guidance on text alone can pin an impact to an exact frame number or make a `<Picture 1>` subject reverse direction mid-shot while keeping its original facing: text tokens carry no frame coordinates. For timed beats and 2D `(x, y, face)` paths, use [`MiniMaxH3SoundCues`](../../../sound_cues.py) and [`MiniMaxH3AppendSpriteTrajectory`](../../../sprite_trajectory.py) with `0.33` selective release on `MiniMaxH3ReleaseReferences` ([guide](research/2026-11-12_sound_cues_dpad_and_033_release_law.md)).
- So the choice on prompt-level steering is between two tools. The word readout is what you use to look at a
  vector, clean it, or make a quick one.

---

## 2. What to call the word readout

Inside the repo it is "J-Lens". In anything written for other people, do not use that name bare.

"J-lens" is the name of a published method from Anthropic, with properties its authors tested. The reading
notes are in
[`research/2026-11-12_chain_lens_refs/jlens_workspace_paper.md`](research/2026-11-12_chain_lens_refs/jlens_workspace_paper.md).
Ours shares its core idea and differs in ways that matter to a reader who knows the paper:

- **It reads steering directions.** The published lens reads a language model's hidden states to see what the
  model is about to say. Ours reads a control vector.
- **It sits in a text encoder inside a video model.** The upper layers it stands in for never run when H3
  makes a video. The video model takes the encoder's state part of the way up and discards the rest.
- **It adds two steps of its own.** One carries a vector back through the token refiner. The other scores
  words against a few fill-in-the-blank prompts.
- **It was fitted on far less data than the published recipe.** Compare section 2 of
  [`research/2026-11-06_j_lens_and_refiner_jacobian.md`](research/2026-11-06_j_lens_and_refiner_jacobian.md)
  with section 1 of the reading notes above.
- **The code that fitted its word half is not in the repo.** Section 17 of
  [`research/2026-11-09_control_vector_simplification_plan.md`](research/2026-11-09_control_vector_simplification_plan.md)
  records that, and rebuilding it is an open item.

### Wording to use

- **In full:** "a Jacobian-lens word readout for H3 control vectors, adapted from Anthropic's Jacobian lens
  (J-lens)".
- **For reading and scrubbing:** "the word readout".
- **For building a vector from words:** "a word vector".

Keep the two uses apart. "The J-Lens" has meant both in this repo, and they answer different questions: what
is this vector about, and make me a vector about this.

### Say plainly

- It is a linear approximation. It is not a trained model.
- It reports what the text encoder would have said next. It does not report what the DiT does with a vector.
- Only a render shows what a vector does.

---

## 3. What can be said with a source

Each of these has a dated record. Give the record with the claim, and do not copy its figures into prose.

- **Its words name most bank vectors, even against their nearest neighbors.** A judge who saw only a vector's
  words picked that vector's concept out of five near concepts for most of the bank. A random direction's
  words were almost never picked.
  [`../../../bench/results/2026-11-12_control_vector_readout_blind_score.md`](../../../bench/results/2026-11-12_control_vector_readout_blind_score.md),
  sections 7 and 8.
- **It cannot tell some close neighbors apart.** The misses sit in a few tight groups, such as old video looks.
  The same record, sections 7.3 and 9.2.
- **A vector's own words point back at it, in part.** A word vector built from a vector's own readout is far
  closer to that vector than to any other, and still some way from it.
  [`../../../bench/results/2026-11-12_control_vector_word_share.md`](../../../bench/results/2026-11-12_control_vector_word_share.md),
  section 3.
- **Rebuilding the bundle moved the words and not the score.** The blind-score record, sections 8.4 and 9.

### What has no source yet

- **Whether the part the words name is the part the DiT responds to.** Clips exist and wait for the owner's
  eye. The word-share record, section 6.
- **Whether a second judge model agrees.** Every pass so far used one.
