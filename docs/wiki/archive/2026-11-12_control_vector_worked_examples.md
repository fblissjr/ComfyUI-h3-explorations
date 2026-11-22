# Worked Examples: Reading, Checking and Choosing *(Archived 2026-11-21)*

> **Archived / superseded:** Moved out of the active steering wiki on `2026-11-21`. Active hands-on walkthroughs, Pole-A Erasure Q&A, and copy-paste workflows now live in [`../steering.md`](../steering.md), [`../steering/tutorials_progressive_guide.md`](../steering/tutorials_progressive_guide.md), [`../steering/how_pole_a_erasure_works.md`](../steering/how_pole_a_erasure_works.md), and [`../steering/examples_and_recipes.md`](../steering/examples_and_recipes.md).

Last updated: 2026-11-21.

Seven questions a person runs into with control vectors, each with what to run or open and what the record
shows. Every example here rests on a dated record, linked beside it. The figures stay in the records.

For the mechanisms side by side see [`which_tool_and_what_to_call_it.md`](../steering/which_tool_and_what_to_call_it.md), for
**Clamped Pole-A Subspace Erasure (`add_norm_matched_erasure`)** see
[`how_pole_a_erasure_works.md`](../steering/how_pole_a_erasure_works.md), and for the word readout see
[`how_the_word_readout_works.md`](../steering/how_the_word_readout_works.md). For copy-paste render settings see
[`examples_and_recipes.md`](../steering/examples_and_recipes.md).

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
[`building_pairs_and_vectors.md`](../steering/building_pairs_and_vectors.md), section 3, and the 2026-11-11 entries in
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
[`../steering/research/2026-11-06_j_lens_and_refiner_jacobian.md`](../steering/research/2026-11-06_j_lens_and_refiner_jacobian.md)
shows `cursed_cctv`'s words changing once two neighbors were held out of it.

**What it does not show.** Whether a hold makes a vector score better against its neighbors. That use of the
scoring tool is planned and not yet run:
[`../steering/research/2026-11-12_chain_lens_digest_and_experiment_plan.md`](../steering/research/2026-11-12_chain_lens_digest_and_experiment_plan.md),
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

**Do this.** First check why it resisted:

- If the block came from **contradictory natural color or open daylight lighting prose** while steering an opposing
  grade (`noir`, `paper_origami`) in legacy `add_norm_matched`, switch to **`mode = "add_norm_matched_erasure"`**
  (now the `MiniMaxH3Steer` default; see [`how_pole_a_erasure_works.md`](../steering/how_pole_a_erasure_works.md)). On `noir`,
  erasure flipped `5/5` stills and `3/3` `192f` market videos from color to crisp monochrome in one `1x` pass.
- If the problem is **swapping a named object color, exact outfit, or scripted action** while the old noun or action
  stays in the prompt, stop turning strength up and switch to edit guidance: write `old wording -> new wording` in
  `MiniMaxH3PromptEdit` (or `MiniMaxH3AutoEditPrompt`) and set `scale` on `MiniMaxH3EditGuider`.
  [`../edit_guidance.md`](../edit_guidance.md) section 2 is the quick start.

**What the record shows.** On 2026-11-09 a vector could not turn a black mug yellow, at any strength tried,
and could not stop a scripted action. Edit guidance did both.
[`../steering/research/2026-11-10_control_vector_state_history_and_tradeoffs.md`](../steering/research/2026-11-10_control_vector_state_history_and_tradeoffs.md),
section 1. The owner's reading of the 2026-11-11 clothing clips says the same for an exact outfit, and also
says a whole-scene vector can still carry clothing along with a change of world:
[`../decisions.md`](../decisions.md), 2026-11-11.

**What it does not show.** That vectors never move an object. The cost is the other side of the choice: edit
guidance runs the model a second time on every guided step, and the result cannot be saved and reused on
another prompt. When the question is instead **exact frame timing** (landing hops or impacts on specific frames)
or **off-prior 2D trajectories** (a `<Picture 1>` subject stopping mid-shot and walking backward while keeping its
original facing), neither text tool owns frame coordinates—reach for [`MiniMaxH3SoundCues`](../../../sound_cues.py),
[`MiniMaxH3AppendSpriteTrajectory`](../../../sprite_trajectory.py), and `0.33` selective release in
[`../steering/research/2026-11-12_sound_cues_dpad_and_033_release_law.md`](../steering/research/2026-11-12_sound_cues_dpad_and_033_release_law.md).

---

## 6. "Why did one sculpted composite stay warmer than chaining two `MiniMaxH3Steer` nodes?"

**Do this.** When combining a heavy conflicting look (`noir`) with a second strong trait (`chaos_panic`) against a
bright `"late morning ... crates of citrus"` prompt, chain **two separate `MiniMaxH3Steer` nodes** instead of
baking them into one single-vector `sculpt` file. Keep `sculpt` (`COMPOSED_ERASURE_MAX_RANK = 16` in
[`control_vectors.py`](../../../control_vectors.py)) for lightweight single-node blends or baking `hold_words` (`calm_cursed_cctv`).

**What the record shows.** Even after raising `compose_erasure_payload(max_rank=16)` so none of the `8 + 8 = 16`
unioned erasure directions are clipped (`bench/verify_pole_a_erasure.py`), a single RMS-normalized composite at
`strength = 1.5` splits one shared `101.25` RMS shift budget between both traits. In the `2026-11-21` GPU check
(`noir_chaos_composed`, rank-8 then rank-16 math check), fractional allocation stayed warm (`sat_mean = 0.422`),
whereas chaining separate `MiniMaxH3Steer` nodes sharing the same `target` evaluates each vector's full `|s_i| * 67.5`
RMS-normalized `[M, D]` shift independently against unsteered `base_rows64` (`control_vectors.py:3454-3492`) and
sums them in `float64` (`how_pole_a_erasure_works.md`; [`../steering/research/2026-11-20_unpooled_multivector_sliders.md`](../steering/research/2026-11-20_unpooled_multivector_sliders.md), §9).

**What it does not show.** Stacking multiple `scene_only` nodes without disjoint `target` masks can still trip
`STACK_SHIFT_WARN_NORM` (and cross-contaminate prose if `noise`/`chaos` overpower color suppression), so tune each
slider's strength (`noir +1.55`, `chaos_panic +0.90`) and keep acoustic traits disjoint on `soundscape_only`.

---

## 7. "Why did `<Picture 1>`'s pink blazer stay pink under `noir` + `MiniMaxH3ReleaseReferences(0.33)`?"

**Do this.** Decide which gear you want before blaming erasure:

- **Gear 1 (`target = "scene_only"` or `"description_only"`, `end_percent = 1.0`):** Use this when you *want*
  **Selective-Color Subject Lock** — cool silver-grey monochrome room atmosphere while keeping `<Picture 1>`'s exact
  garment colors and identity intact.
- **Full B&W/material conversion on `<Picture 1>`:** Dropping VAE rows at `MiniMaxH3ReleaseReferences(end_percent=0.33)`
  alone is not enough when `MiniMaxH3Steer` protects `retention_analysis:` and Qwen3-VL's `843` vision rows
  (`tags == 0`). Rewrite the color/material clause in `subject_definitions:` / `retention_analysis:` via
  `MiniMaxH3PromptEdit` or `MiniMaxH3AutoEditPrompt`, or apply `target = "all"` plus late ref release cleanly.

**What the record shows.** In `Gate 2B-2` (`3/3` seeds) and `Showcase 3` (`$H3_OUTPUT_DIR/Video/erasure_showcase_2026-11-21/`,
`90f` `<Picture 1>` bookstore), `add_norm_matched` (`s3_ref2va_1d_before`) kept the entire room warm yellow
(`sat_mean = 0.3387`). Both Pole-A Erasure gears (`Gear 1 pin=1.0` `sat_mean = 0.3017`; `Gear 2 rel=0.33`
`sat_mean = 0.3056`) cooled the bookstore walls and shelves to crisp silver-grey with `100%` facial likeness,
oak-book thump, and synced speech while deliberately preserving the bright pink floral blazer encoded in the unsteered
vision tokens (`tags == 0`) and `subject_definitions:` / `retention_analysis:` prose.
