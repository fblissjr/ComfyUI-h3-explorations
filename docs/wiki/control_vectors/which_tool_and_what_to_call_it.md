# Which Tool Is Which, and What to Call the Word Readout

Last updated: 2026-11-12.

This pack has three things that get mentioned together: control vectors, edit guidance, and the J-Lens. They
are not three of a kind. Two of them steer a render. The third mostly reads.

Read this page first if you are explaining the pack to someone else. For the step-by-step mechanism of the
word readout see [`how_the_word_readout_works.md`](how_the_word_readout_works.md). For worked cases see
[`worked_examples.md`](worked_examples.md).

---

## 1. The three, side by side

| | Control vector | Edit guidance | Word readout (J-Lens) |
|---|---|---|---|
| **What it is** | One direction added to the prompt's text tokens | Two prompts compared at every guided step | A translator between a vector and English words |
| **Where it acts** | After the token refiner, before the DiT | Inside the DiT's predicted update | Nowhere in the render, unless its output is wired into a Steer node |
| **Built from** | The average difference between pairs of prompts | Your own prompt and `old wording -> new wording` edits | The text encoder's weights, once, into a bundle file |
| **Render cost** | One model pass, the same as no steering | A second model pass on every guided step | None. It runs on the CPU |
| **Good for** | Reusable looks, moods, weather and sound spaces, on any prompt | An exact change to one outfit, prop or action in one shot | Checking a vector before rendering, scrubbing a side effect, sketching a vector from words |
| **Cannot do** | Reliably replace one specific noun | Be saved and reused on another prompt | Tell you what the video will show |

Where each lives in the code:

- **Control vector**: `control_vector_nodes.py::MiniMaxH3Steer` applies one. The hub page,
  [`../control_vectors.md`](../control_vectors.md), says how they are built and which rows they touch.
- **Edit guidance**: `edit_guidance.py::MiniMaxH3EditGuider` with `edit_guidance.py::MiniMaxH3PromptEdit`.
  [`../edit_guidance.md`](../edit_guidance.md) has the dials and the pass counts.
- **Word readout**: `j_lens.py::decode_vector` reads, `j_lens.py::apply_soft_holds` scrubs, and
  `j_lens.py::synthesize_from_words` builds. The node that builds is
  `control_vector_nodes.py::MiniMaxH3WordVector`.

### How they relate

- A vector made by the word readout **is** a control vector. It goes through the same Steer node. The readout
  is a second way to make one, from a word list in place of prompt pairs.
- Edit guidance uses no vector and no readout. It is a different mechanism with a different cost.
- So the choice a person faces is between two steering tools. The word readout is what you use to look at a
  vector, clean it, or make a quick one.

---

## 2. What to call the word readout

Inside the repo it is "J-Lens". In anything written for other people, do not use that name bare.

"J-lens" is the name of a published method from Anthropic, with properties its authors tested. The reading
notes are in
[`../../research/control_vectors/2026-11-12_chain_lens_refs/jlens_workspace_paper.md`](../../research/control_vectors/2026-11-12_chain_lens_refs/jlens_workspace_paper.md).
Ours shares its core idea and differs in ways that matter to a reader who knows the paper:

- **It reads steering directions.** The published lens reads a language model's hidden states to see what the
  model is about to say. Ours reads a control vector.
- **It sits in a text encoder inside a video model.** The upper layers it stands in for never run when H3
  makes a video. The video model takes the encoder's state part of the way up and discards the rest.
- **It adds two steps of its own.** One carries a vector back through the token refiner. The other scores
  words against a few fill-in-the-blank prompts.
- **It was fitted on far less data than the published recipe.** Compare section 2 of
  [`../../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`](../../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md)
  with section 1 of the reading notes above.
- **The code that fitted its word half is not in the repo.** Section 17 of
  [`../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md)
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
