# Worked Examples: Reading, Checking and Choosing

Last updated: 2026-11-12.

Five questions a person runs into with control vectors, each with what to run or open and what the record
shows. Every example here rests on a dated record, linked beside it. The figures stay in the records.

For the tools themselves see [`which_tool_and_what_to_call_it.md`](which_tool_and_what_to_call_it.md) and
[`how_the_word_readout_works.md`](how_the_word_readout_works.md). For copy-paste render settings see
[`examples_and_recipes.md`](examples_and_recipes.md).

---

## 1. "Will this vector do what I built it for?"

**Do this.** Read its words before you render. Open the `.md` file that sits beside the vector in
`<comfy>/models/control_vectors/`, or run:

```bash
CUDA_VISIBLE_DEVICES= <comfy-venv-python> bench/control_vectors.py inspect <vector file>
```

Read the `j_lens (PR +)` line. Ask whether the words name the thing you wanted, or only its mood.

**What the record shows.** A vector called `wardrobe_1970s` was built from six different 1970s outfits. Its
words came back as an era ("vintage", "psychedelic", "nostalgic") with no garment among them. On playback it
tinted the room and left the subject's sweater alone. Nobody had read the words first.
[`building_pairs_and_vectors.md`](building_pairs_and_vectors.md), section 3, and the 2026-11-11 entries in
[`../decisions.md`](../decisions.md).

**What it does not show.** Good words do not promise a good render. They catch one kind of failure early: a
vector that carries something other than what it was built for.

---

## 2. "Do the words really name the vector, or do they just sound right?"

**Do this.** Score the readouts blind. A judge sees only a vector's words and has to pick its concept from
five near concepts. The tool builds the judge's packets and counts the answers; the judging is done by a model
you hand the packets to.

```bash
<comfy-venv-python> bench/control_vector_readout_score.py packets --out data/control_vectors/<run> \
    --rows data/control_vectors/<rows cache> --tokenizer <comfy>/comfy/text_encoders/qwen25_tokenizer
<comfy-venv-python> bench/control_vector_readout_score.py score data/control_vectors/<run>
```

The docstring of `bench/control_vector_readout_score.py` says how a run is built and where the answers go.
Give a judge one packet and the instructions, never the run folder: the answers are in it.

**What the record shows.**

- The words named most of the bank's vectors, even against their nearest neighbors.
- A random direction's words were almost never taken for a real vector's. That is the control that says the
  hits are not luck or leading options.
- A second set of judges on the same items agreed almost item for item.
- Rebuilding the bundle changed the words for every vector and left the count where it was.

[`../../../bench/results/2026-11-12_control_vector_readout_blind_score.md`](../../../bench/results/2026-11-12_control_vector_readout_blind_score.md),
sections 7, 8 and 9.

**What it does not show.** It scores naming, not effect. One judge model was used throughout. And a miss can
be the vector's fault as much as the readout's: see the next example.

---

## 3. "Why does my film vector read like the VHS one?"

**Do this.** Look at which vectors sit close to yours. The `packets` command above prints each group of near
vectors and how close its members are. If two vectors overlap, their words will too.

**What the record shows.** With the bundle as it stood before 2026-11-12, `film_16mm_1974`'s words included
"tape", and blind judges picked the VHS vector for it. `cursed_cctv` went the same way. The three sit in one
tight group of old video looks. After the bundle was rebuilt, "tape" was gone from the film vector's words and
judges named it; `cursed_cctv` was still taken for VHS.
[`../../../bench/results/2026-11-12_control_vector_readout_blind_score.md`](../../../bench/results/2026-11-12_control_vector_readout_blind_score.md),
sections 6.4, 7.3 and 9.2.

A hold can pull two such vectors apart. Section 5.3 of
[`../../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`](../../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md)
shows `cursed_cctv`'s words changing once two neighbors were held out of it.

**What it does not show.** Whether a hold makes a vector score better against its neighbors. That use of the
scoring tool is planned and not yet run:
[`../../research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md`](../../research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md),
section 13.

---

## 4. "What is a vector actually made of?"

**Do this.** Open the record. It splits a vector into three parts at right angles to each other:

- the direction every bank vector shares,
- the direction the vector's own words name beyond that,
- and the rest.

**What the record shows.**

- Every bank vector carries a sizeable share of one common direction. Each pair file sets its concept against
  something plain, so every vector has some "away from plain" in it.
- A word vector built from a vector's own readout points at that vector far more than at any other. It is
  still only part of the vector.
- An earlier way of asking the question, fitting a vector with a handful of word directions, turned out to
  fit a random direction just as well. It is in the record as a method not to use.

[`../../../bench/results/2026-11-12_control_vector_word_share.md`](../../../bench/results/2026-11-12_control_vector_word_share.md),
sections 2, 3 and 3.1.

**What it does not show yet.** Which part the video model responds to. Eight short clips were rendered for
that, one for each part of two vectors plus an unsteered base, under
`$H3_OUTPUT_DIR/Video/cv_word_share_2026-11-12/`. They are waiting for the owner's eye and have not been
judged. Section 6 of the same record says how they were made and what to look at.

---

## 5. "The vector will not change this one thing."

**Do this.** Stop turning the strength up and switch tools. A control vector is one direction spread over many
prompt rows. It is good at a look, a mood or a kind of weather. It is poor at swapping one named thing for
another while the old word is still in the prompt.

Use edit guidance: write the change as `old wording -> new wording` in `MiniMaxH3PromptEdit`'s `edits`, and
set `scale` on `MiniMaxH3EditGuider`. [`../edit_guidance.md`](../edit_guidance.md) section 2 is the quick
start.

**What the record shows.** On 2026-11-09 a vector could not turn a black mug yellow, at any strength tried,
and could not stop a scripted action. Edit guidance did both.
[`../../research/control_vectors/2026-11-10_control_vector_state_history_and_tradeoffs.md`](../../research/control_vectors/2026-11-10_control_vector_state_history_and_tradeoffs.md),
section 1. The owner's reading of the 2026-11-11 clothing clips says the same for an exact outfit, and also
says a whole-scene vector can still carry clothing along with a change of world:
[`../decisions.md`](../decisions.md), 2026-11-11.

**What it does not show.** That vectors never move an object. The cost is the other side of the choice: edit
guidance runs the model a second time on every guided step, and the result cannot be saved and reused on
another prompt. When the question is instead **exact frame timing** (landing hops or impacts on specific frames)
or **off-prior 2D trajectories** (a `<Picture 1>` subject stopping mid-shot and walking backward while keeping its
original facing), neither text tool owns frame coordinates—reach for [`MiniMaxH3SoundCues`](../../../sound_cues.py),
[`MiniMaxH3AppendSpriteTrajectory`](../../../sprite_trajectory.py), and `0.33` selective release in
[`../../research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md`](../../research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md).
