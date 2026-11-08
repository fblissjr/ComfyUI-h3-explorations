# 7. Prompting Generative LLMs for Control Vectors (Input Templates & Examples)

**Parent guide**: [`docs/wiki/control_vectors.md`](../control_vectors.md)
**Canonical System Prompt**: [`docs/wiki/system_prompts/h3_control_vector_system_prompt.md`](../system_prompts/h3_control_vector_system_prompt.md)
**Related guides**: [`2. Building Pairs & Vectors`](building_pairs_and_vectors.md), [`6. Practical Examples & Recipes`](examples_and_recipes.md)

---

## 1. How the Generative LLM System Prompt Works

The canonical system prompt in [`docs/wiki/system_prompts/h3_control_vector_system_prompt.md`](../system_prompts/h3_control_vector_system_prompt.md) can be used in **two ways**:
1. **Inside our Studio UI or CLI**:
   - **Studio UI (`Screen 3: Full Pair & LLM Studio`)**: Select a supported local provider (`transformers` or `exllamav3`), enter your local model path or HuggingFace model ID, and type your user prompt into the **Concept / Axis Description** box.
   - **CLI (`generate-pairs`)**:
     ```bash
     COMFY="$(cd ../.. && pwd)"
     PY="$COMFY/.venv/bin/python"

     "$PY" bench/control_vectors.py generate-pairs \
       --llm-provider exllamav3 \
       --llm-model /path/to/your/exl3_model \
       --concept "<YOUR USER PROMPT HERE>" \
       --name my_vector_name \
       --save-toml \
       --overwrite
     ```
2. **In Any External Generative LLM Service (Chat UI, API, or Local Server)**:
   - Copy the fenced block from [`docs/wiki/system_prompts/h3_control_vector_system_prompt.md` §1](../system_prompts/h3_control_vector_system_prompt.md#1-copyable-system-prompt) (or click **Copy System Prompt** in **Studio Screen 3**) and set it as the **System Prompt** (or first message).
   - Send your **User Prompt** using one of the structured patterns below.
   - Paste the LLM's output into **Studio Screen 3** (or save as `my_vector.md` and run `"$PY" bench/control_vectors.py build my_vector.md --name my_vector --overwrite`).

---

## 2. What Makes a Great User Input Prompt for the LLM?

Because [`h3_control_vector_system_prompt.md`](../system_prompts/h3_control_vector_system_prompt.md) instructs the LLM to produce a **Test Scene Prompt** plus **10 Stratified Contrastive Pairs (`Prompt A` vs. `Prompt B`)** obeying strict single-clause surgical substitution (`MAX_SHARED_INTERIOR_WORDS = 5`) and anti-cancellation rules ($B - A$), vague one-word prompts (like `"make a cool ocean vector"`) leave too much ambiguity.

To get the best output on the first try, specify **four elements** in your user prompt:
1. **Output Mode Requested**: State whether you want:
   - **Mode 1 (Default)**: A full **10-Pair Control Vector Set** (`### Test Scene Prompt` + `### Contrastive Pairs (1–10)` + optional `### Hold Pairs`), or
   - **Mode 2**: A compact **Short-Contrast Compiler Spec (`build --words`)** for `h3_pair_compiler.py`.
2. **Pole A (Baseline State) vs. Pole B (Steered Target State)**:
   - State what `Prompt A` should describe (the ordinary unsteered baseline, **without** mentioning any of `Pole B`'s target nouns so $B - A$ never cancels out) and what `Prompt B` should replace that single clause with.
3. **Whether Audio (`overall_soundscape:`) Should Also Shift or Stay Identical**:
   - Say **either**: *"Change ONLY the visual description clause; keep `overall_soundscape:` 100% identical between Prompt A and Prompt B"* **or** *"Use a coupled `description + overall_soundscape` diff where one visual clause and one soundscape clause change in each pair."*
4. **Your Test Scene Scenario (and optional `Hold` Confounds to Project Out)**:
   - Describe the unsteered scene you want in `### Test Scene Prompt` (and whether `Test Scene Prompt` should be `t2va` 3-section or `ref2va` 6-section), plus any nuisance confound (`### Hold Pairs`) to scrub out.

---

## 3. Copy-Paste User Prompt Examples (By Vector Category)

### Category A — Environment, Lighting & Optical Atmosphere (Visual + Soundscape Coupled)

Use this pattern when steering lighting, weather, underwater/cavern atmosphere, or film stock across the whole scene.

#### Input Prompt Example A1: Bioluminescent Deep-Sea Cavern (`10-Pair Control Vector Set`)
```text
Build a 10-Pair Control Vector Set (### Test Scene Prompt + ### Contrastive Pairs (1–10)).

- Concept / Axis: Ordinary sunlit or neutrally lit everyday environments (Pole A) -> Flooded with glowing cyan-and-violet bioluminescent cavern flora, drifting luminous spore particles, and caustic cyan light ripples (Pole B).
- Diff Rule: Use a single contiguous clause diff in `integrated_multimodal_description:` (8–16 words) AND a coupled single contiguous clause diff in `overall_soundscape:` (ordinary room/outdoor ambient air -> resonant subterranean cavern water drips and low crystalline hum). Never put words like "cavern", "bioluminescent", "spores", or "cyan" in Prompt A.
- Test Scene Prompt (t2va): A lone watchmaker at a wooden workbench examining a brass pocket watch and saying a short 4-word line in English. Keep the Test Scene Prompt 100% unsteered.
```

#### What This Produces (Expected Output Structure)
- `### Test Scene Prompt`: Unsteered 3-section `t2va` watchmaker prompt with `<d>[English] ...</d>` (`1–6` words) and zero mention of caverns or bioluminescence.
- `### Contrastive Pairs (1–10)`: 5 indoor + 5 outdoor unrelated scenes (none involving a watchmaker), mixing speaking `(S1)`, silent human (`produces no vocal sound`), and environment-only (`No people are present in the scene.`) shots, where each pair changes exactly one visual lighting/environment clause and one soundscape clause.

---

### Category B — Wardrobe, Material, or Character Appearance Override (`ref2va`-Ready)

Use this pattern when you want a control vector that changes what a character is wearing or made of—including overriding `<Subject 1>`'s outfit in `ref2va` (applied on `start_percent = 0.0, end_percent = 0.50` at `strength = 2.0–2.5` with `ref_vae_end_percent = 1.00` in ComfyUI).

#### Input Prompt Example B1: Everyday/Winter Attire $\rightarrow$ Tropical Summer Beachwear (`10-Pair Control Vector Set` + `Hold`)
```text
Build a 10-Pair Control Vector Set (### Test Scene Prompt + ### Contrastive Pairs (1–10) + ### Hold Pairs).

- Concept / Axis: Subjects dressed in ordinary everyday indoor or cool-weather street clothing such as sweaters, jackets, aprons, or button-down shirts (Pole A) -> Subjects dressed in bright summer beachwear, lightweight linen resort wear, and tropical swimwear (Pole B).
- Diff Rule: Change ONLY one contiguous wardrobe/attire clause (8–15 words) inside `integrated_multimodal_description:`. Keep `overall_soundscape:` and `non_diegetic_music: N/A` 100% character-for-character identical between Prompt A and Prompt B so the room and environment do not change.
- Anti-Cancellation Rule: Never mention beachwear, swimwear, bikini, trunks, or summer clothing in Prompt A.
- Hold Pairs (2 pairs, group `environment`): Add 2 Hold Pairs that contrast an indoor room vs. an outdoor sandy beach while keeping clothing constant, so projecting out `hold(environment)` ensures the wardrobe vector changes only clothing without turning indoor rooms into beaches.
- Test Scene Prompt (ref2va 6-section format): <Subject 1> from <Picture 1> standing in a modern kitchen preparing iced coffee and saying a short 4-word greeting in English.
```

#### Why This Input Works So Well
1. **Broad Pole A (`sweaters, jackets, aprons, or button-down shirts`)**: Ensures the extracted direction $B - A$ replaces *any* everyday clothing rather than requiring the target prompt to mention winter coats.
2. **Visual-Only Diff (`overall_soundscape:` identical)**: Prevents ocean waves from bleeding into indoor kitchen audio.
3. **`Hold Pairs (environment)`**: Explicitly tells the LLM to include 2 `#### Hold 1 (environment)` pairs contrasting indoor vs. beach settings so `make_control_vector_from_pairs` orthogonalizes clothing against room background bleed.
4. **6-Section `ref2va` `Test Scene Prompt` + 3-Section `Contrastive Pairs`**: Matches §1.2 of the system prompt (`Contrastive Pairs` always use 3-section `integrated_multimodal_description:` so one build emits both `fl2va` and `ref2va` vectors, while `Test Scene Prompt` uses 6-section `ref2va`).

---

### Category C — Physical Dynamics, Zero Gravity, or Object Behavior (Post-Speech Action Beat)

Use this pattern when steering physics, levitation, slow-motion impact, or object transformation.

#### Input Prompt Example C1: Sudden Zero-Gravity Levitation (`10-Pair Control Vector Set`)
```text
Build a 10-Pair Control Vector Set (### Test Scene Prompt + ### Contrastive Pairs (1–10)).

- Concept / Axis: Loose scene props and droplets resting normally on surfaces under standard gravity (Pole A) -> Loose scene props, tools, and liquid droplets lifting smoothly off surfaces and drifting weightlessly in mid-air (Pole B).
- Diff Rule: Place the shared scene prop nouns in the identical sentence prefix outside the diff clause, and change ONLY one contiguous 8–16 word physical-state clause in `integrated_multimodal_description:`. On speaking pairs (S1), place the differing clause after the speaker finishes their short `<d>[English] ...</d>` line and closes their lips.
- Banned Words Reminder: Do not use "slowly", "gradually", "steadily", "begins to", or "starts to", and never write the word "gravity" or "weightless" in Prompt A.
- Test Scene Prompt (t2va): A barista wiping an espresso counter with ceramic cups and spoons beside the machine after saying a 3-word line in English.
```

---

### Category D — Camera Motion & Cinematography (`opening_framing` / `camera_motion`)

Use this pattern when isolating a pure camera movement or lens behavior without changing subject action or lighting.

#### Input Prompt Example D1: Aggressive Handheld Camcorder Shake (`10-Pair Control Vector Set`)
```text
Build a 10-Pair Control Vector Set (### Test Scene Prompt + ### Contrastive Pairs (1–10)).

- Concept / Axis: The camera holds a locked-off, tripod-stable frame with smooth composed framing (Pole A) -> The camera shakes strongly with frantic handheld camcorder urgency, abrupt whip-pans, and erratic framing jolts (Pole B).
- Diff Rule: Change ONLY the single contiguous camera-behavior clause in `integrated_multimodal_description:` across 5 indoor and 5 outdoor scenes, varying the clause position across opening, mid-shot, and closing sentences. Keep `overall_soundscape:` identical between Prompt A and Prompt B.
- Test Scene Prompt (t2va): Two fencers saluting each other on a gymnasium strip.
```

---

### Category E — Compact `--words` Compiler Spec (`Mode 2` of the System Prompt)

When you want the LLM to write a compact **Short-Contrast Compiler Spec** for [`h3_pair_compiler.py`](../../../h3_pair_compiler.py) (which expands into 10 canonical scenes in `< 1 ms` on CPU using scene-adaptive placeholders `{props}`, `{surface}`, `{poss}`, `{obj}`) instead of writing 10 full Markdown pairs:

#### Input Prompt Example E1: Asking the LLM for a `--words` Spec
```text
Write a Short-Contrast Compiler Spec (`build --words` format for `h3_pair_compiler.py`, NOT 10 full Markdown pairs) for a "Cryogenic Flash-Freeze" control vector:
- Use slot `subject_behavior` with `{props}` and `{surface}` placeholders so each canonical scene's native objects freeze over with creeping blue rime frost and ice crystals.
- Include a coupled `soundscape:` clause shifting ambient room tone into sharp crystalline ice-cracking and sub-zero wind hiss.
- Include a `hold(wardrobe):` clause so actors' indoor clothing does not turn into arctic parkas.
```

#### Expected LLM Output (Single Copy-Paste Spec Line)
```text
subject_behavior: while {props} rest undisturbed on {surface} at normal room temperature -> while {props} across {surface} flash-freeze under creeping blue rime frost and branching ice crystals; soundscape: quiet ambient room tone -> sharp crystalline ice cracking and sub-zero frost hiss; hold(wardrobe): dressed in indoor casual shirt sleeves -> bundled in a heavy insulated arctic parka and fur hood
```

You can pass that output line directly to:
```bash
"$PY" bench/control_vectors.py build \
  --words "subject_behavior: while {props} rest undisturbed on {surface} at normal room temperature -> while {props} across {surface} flash-freeze under creeping blue rime frost and branching ice crystals; soundscape: quiet ambient room tone -> sharp crystalline ice cracking and sub-zero frost hiss; hold(wardrobe): dressed in indoor casual shirt sleeves -> bundled in a heavy insulated arctic parka and fur hood" \
  --name cryogenic_freeze \
  --save-toml \
  --overwrite
```

---

## 4. Pre-Flight Checklist When Reviewing LLM Outputs

Before clicking **Build Vector** in the Studio UI (or running `bench/control_vectors.py build`), check these four items (the Studio UI's **Validate & Preview Token Diffs** button checks items 1–3 automatically, and our LLM pipeline auto-repairs minor secondary edits):

| Check | What to Look For | How the Studio / CLI Helps |
|---|---|---|
| **1. One Contiguous Clause Per Section** | `Prompt A` and `Prompt B` differ in only one contiguous clause in `integrated_multimodal_description:` (plus optionally one contiguous clause in `overall_soundscape:`). | `bench/control_vector/llm_pairs.py::_repair_multichange_markdown_pairs` automatically repairs secondary non-contiguous edits in a section, and **Validate & Preview Token Diffs** highlights the exact `diff_a -> diff_b` span for all 10 pairs. |
| **2. Zero Target-Noun Overlap ($B - A$)** | None of the target effect keywords (`cavern`, `frost`, `swimwear`, `VHS`) appear in `Prompt A`'s diff span. | Check the `-diff_a -> +diff_b` preview badges in **Studio Screen 3** to verify `diff_a` describes only the neutral baseline state. |
| **3. `non_diegetic_music: N/A` Identical** | Unscored pairs use `non_diegetic_music: N/A` on both `Prompt A` and `Prompt B`. | Keeps music tokens out of `TokenRefiner` unless you are specifically building a music control vector. |
| **4. `10/10` Unique Diff Spans** | Each pair `Pair 1..10` uses distinct wording rather than copy-pasting the same clause 10 times. | Averaging 10 distinct phrasings gives high `split_half_cosine` (`>= 0.75`) that generalizes to unseen prompts. |
