# Using the standalone control-vector prompt: examples

Last updated: 2026-11-09.

Examples for [`h3_control_vector_standalone_system_prompt.md`](h3_control_vector_standalone_system_prompt.md):
what to send it, roughly what comes back, what a good request looks like, and where this approach fits for
the two video model families.

Two things to know first:

- The prompt has not been run on a model yet. Every "expected output" below was written by hand to show the
  shape, not copied from a model. The first real run is a throwaway to see what a model actually does with it.
- The generator (`bench/control_vector/llm_pairs.py`) does not read the standalone file yet. Until it does,
  use it by pasting: the whole file as the system prompt, your request as the user message.

## 1. The two things you can ask for

| You want | Ask for | You get back |
|---|---|---|
| A reusable look, built from many scenes | a pair set | a test scene prompt, then 10 prompt pairs, each pair identical but for one clause |
| A change to one prompt you already have | an edit list | one line per edit: exact text from your prompt, an arrow, the replacement |

The pair set is the slow road: ten pairs to read, then a build. The edit list is one or two lines and feeds
the in-place method directly. If you only want one shot changed, ask for the edit list, or just rewrite the
prompt.

## 2. Example: a pair set

### What you send

```text
Write a pair set.

Quality: lit only by candle flame, in place of whatever light the scene has.
Sound: do not change the soundscape.
Test scene: a woman at a kitchen table reads a letter and says "He's coming home."
```

### Roughly what comes back

One fenced block holding the test scene, then ten pairs. Two pairs are shown here; the other eight follow
the same pattern in other places.

```text
### Test Scene Prompt

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames a small kitchen where a woman in a grey cardigan, on-screen, forties, with a low, steady alto (S1), sits at a wooden table holding an opened letter. She reads to the last line, looks up toward the window and says: <d>[English] He's coming home.</d> She closes her mouth and folds the letter along its crease. The camera holds a static shot.

overall_soundscape: A refrigerator hums, paper creases under her fingers, and a kettle ticks as it cools on the stove.

non_diegetic_music: N/A

### Contrastive Pairs (1–10)

#### Pair 1
##### Prompt A (Baseline)
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames a narrow bookbinding workshop where two ceiling tubes cast flat white light across the cutting bench and paper stacks. A lone bookbinder in a canvas apron, on-screen, sixties, with a dry, quiet baritone (S1), presses a spine flat with a bone folder and says: <d>[English] Almost square now.</d> He closes his mouth and runs his thumb along the edge. The camera holds a static shot.

overall_soundscape: A bone folder scrapes along board, loose pages shift, and a clock ticks on the far wall.

non_diegetic_music: N/A

##### Prompt B (Steered)
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames a narrow bookbinding workshop where three stub candles throw wavering amber light across the cutting bench and paper stacks. A lone bookbinder in a canvas apron, on-screen, sixties, with a dry, quiet baritone (S1), presses a spine flat with a bone folder and says: <d>[English] Almost square now.</d> He closes his mouth and runs his thumb along the edge. The camera holds a static shot.

overall_soundscape: A bone folder scrapes along board, loose pages shift, and a clock ticks on the far wall.

non_diegetic_music: N/A

#### Pair 2
##### Prompt A (Baseline)
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide shot frames a ferry ramp at dusk, sodium lamps on tall poles washing the wet concrete in even orange. A lone deckhand in a yellow jacket coils a rope beside the bollard from the left of the frame toward the center, his lips closed, and produces no vocal sound. The camera pans right at slow speed.

overall_soundscape: Water slaps the ramp, a rope drags across concrete, and a chain clinks against the bollard.

non_diegetic_music: N/A

##### Prompt B (Steered)
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide shot frames a ferry ramp at dusk, jar candles along the low railing flickering over the wet concrete in uneven gold. A lone deckhand in a yellow jacket coils a rope beside the bollard from the left of the frame toward the center, his lips closed, and produces no vocal sound. The camera pans right at slow speed.

overall_soundscape: Water slaps the ramp, a rope drags across concrete, and a chain clinks against the bollard.

non_diegetic_music: N/A
```

### What to check in the output before building from it

- Each pair differs in one clause only. Everything else matches character for character.
- The plain clause names what is really in that scene (ceiling tubes, sodium lamps). It does not say
  "natural light" or "ordinary lighting" ten times.
- The two clauses are about the same length.
- No candle, flame or wick appears on the plain side.
- No two pairs share a clause, and the test scene's kitchen, letter and line appear in none of them.
- The test scene has no candle in it and no sentence insisting the light stays as it is.

## 3. Example: an edit list for a reference prompt

### What you send

Your finished prompt, and the change. This one is a ref2va prompt with one reference picture.

```text
Write an edit list for this prompt. Quality: a 1990s multi-camera studio sitcom, seen and heard.

subject_definitions:
<Subject 1> is the young woman in <Picture 1>, with very long, loose, wavy blonde hair parted in the center and falling past her chest in sun-streaked strands, blue eyes, and light freckles across her nose and cheeks.

summary:
[reference generation] <Subject 1> stands inside a sunlit diner and speaks to the camera.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - her facial features, long wavy center-parted blonde hair, blue eyes, light freckles, and overall identity from <Picture 1> are completely retained.

detailed_description:
Live-action footage with natural cinematic lighting. [Shot 1] Inside a sunlit diner, <Subject 1> stands in front of a table. The camera holds a medium shot as <Subject 1> (S1) says with an amused expression in a warm, amused voice, <d>[English] Want more pie?</d> before settling her lips.

overall_soundscape:
Atmospheric diner sounds and faint murmurs of distant patrons.

non_diegetic_music:
N/A
```

### Roughly what comes back

```text
Live-action footage with natural cinematic lighting. -> Multi-camera 1990s studio sitcom footage under flat, bright stage lighting.
soundscape: Atmospheric diner sounds and faint murmurs of distant patrons. -> A studio audience laughs warmly over the low hum of a soundstage.
```

### What to check

- The text left of each arrow is in your prompt exactly once. "amused" would be a bad choice here: it occurs
  twice.
- Nothing inside `<d>...</d>`, no tag and no section header is on either side.
- Each replacement is about as long as what it replaces.

## 4. Example: an edit list for a text-only prompt

### What you send

```text
Write an edit list for this prompt. Quality: claymation stop-motion. Picture only.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames a late-night beef-noodle counter from the street side, a steel stockpot steaming under a bare bulb and a row of red plastic stools along the tiled front. (the rest of the prompt, pasted whole)
```

### Roughly what comes back

```text
Live-action, cinematic, -> Claymation stop-motion with fingerprinted plasticine surfaces,
```

## 5. Good requests and bad ones

A request works when it says four things: which output you want, the quality as a change from something to
something, whether sound changes too, and (for a pair set) the scene you will test on.

| Request | What is wrong or right with it |
|---|---|
| "Make a cool noir vector." | Bad. It names a label and nothing else. The model has to guess what the plain side is, whether the sound changes, and what scene you care about. |
| "Pair set. Quality: high-contrast black-and-white with hard shadow edges, in place of the scene's own color and light. Picture only. Test scene: a night porter in a hospital lift says 'One more round.'" | Good. One quality, stated as a change; sound settled; a test scene given. |
| "Pair set for rain and a handheld camera and a whispering voice." | Bad. Three qualities in one request. The pairs can only change one clause, so ask three times and combine the results afterward. |
| "Edit list: make her say 'Get out' instead." | Bad use. That changes the dialogue, which is not a quality. Edit the line in your prompt yourself. |
| "Edit list for this prompt: make it feel better." | Bad. No quality to write a clause for. |
| "Edit list for this prompt. Quality: heavy fog, seen and heard." with the whole prompt pasted | Good. The prompt is there to copy from, and "seen and heard" tells it to write a soundscape line too. |
| A pair-set request that pastes your own scene as the place for all ten pairs | Bad. The pairs must be ten unrelated scenes; your scene belongs only in the test scene line. |

## 6. Where this fits, by model family

Pairs are always written as text-only prompts, whichever family the vector is built for. The family decides
how the result behaves, not how you ask.

What follows leans on the arms in
[`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md).
Each arm there is one clip at one seed, read by a session, so treat these as indications and not as settled.
Anything not listed there is untested.

### fl2va (text-to-video and keyframe video)

| Use | Fit | Why |
|---|---|---|
| A film stock, a grade, a lighting look | Good | The record's film-look and weather arms changed the look of the whole frame. |
| Weather and atmosphere, with their sound | Good | The rain and wind arm changed picture and sound together. |
| A vocal delivery | Good | The operatic arm changed how the line was delivered and kept the words. |
| A material or animation style for the whole scene | Works, with side effects | The claymation arm changed the material and also restaged the shot. Expect composition to move. |
| Keeping the exact framing while changing the look | Use a late start | The record's late-window arm kept the layout and changed only the look. From the first step, a strong change also moves the layout. |
| Placing an object at an exact spot, or matching a silhouette | Bad | Nothing here knows about position. |
| Changing what is said | Bad | Dialogue is kept apart on purpose. |
| A change you want on one shot only | Use the edit list, or rewrite the prompt | Ten pairs are for a look you will reuse. |

### ref2va (video from a reference picture)

| Use | Fit | Why |
|---|---|---|
| Changing the world around the subject: the look, the era, the weather | Good | In the record's ref2va arms the street and the light changed while the person from the picture stayed the same. |
| Physical effects on things in the scene | Good | The arms with objects lifting and with smoke changed what happened while identity held. |
| Changing how the subject speaks or moves | Plausible | Delivery changed in the ref2va arms that had speech; fewer arms cover it. |
| Turning the referenced subject into another material (felt, rubber, clay) | Poor | The reference picture holds the subject's appearance. In the record's arms the scene changed and the subject stayed a photographed person. |
| Changing the subject's identity, face or hair | Bad | That is what the reference picture is for. Change the picture. |
| Changing the subject's clothing | Hard | It works against the reference. If you try it, the reference text must not describe the old clothing, and the retention line becomes `partially_preserved`. |

## 7. When not to use this prompt at all

- You have not tried editing the prompt directly. A direct edit is the baseline every other method competes
  with.
- The quality cannot be seen or heard in a clause: a mood with no visible form, a plot point, a name.
- You want a strength dial on one shot. The edit list plus the edit-guidance method under test in
  [`2026-11-09_control_vector_simplification_plan.md`](../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md)
  (section 11) is the shorter road, if that test goes well.
