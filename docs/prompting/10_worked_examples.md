> **Part of [`docs/prompting.md`](../prompting.md) (Graded Worked Examples (`T2VA`, `I2VA`, `FL2VA`, `L2VA`, `ref2va`, §10)).**
> For the core prompt structure (`§1`–`§5.7`) and master router, see [`docs/prompting.md`](../prompting.md).

## 10. Worked examples

**These are mine, not the guides'.** They are written to this manual and are
conformant against every GUIDE rule above; where a HOUSE rule is contested I say
so rather than pretending the example settles it. The guides' own worked examples
are base §5 Cases 1-4 and ref §7.

**A broader graded set lives in [`prompt_bank/`](../../prompt_bank/)**, catalogued
in [`prompt_bank.md`](../prompt_bank.md) by `bench/build_prompt_bank.py`, whose
`--check` re-grades every prompt there through the same function
`grade_prompt_text.py` runs. It exercises every value of the closed sets this
manual lists and says which it cannot. Those are conformant drafts and probe
prompts written to the same layers as this section; they are neither this
manual's worked examples nor the vendor's, and the top of that file says so
before anything else.

**Every example below is GRADED, and a check re-runs that grading.** Until
2026-09-01 this section asserted that its examples "grade clean through
`preflight_graph.py`" and nothing checked it; `bench/check_prompt_docs_sync.py`
now re-grades every one at the duration its heading names, so an edit that
breaks an example goes red instead of shipping as the reference for "good" -- preflight reads graphs, and
these are loose text. `bench/grade_prompt_text.py` closes that: it wraps a
prompt in a shipped graph of the requested mode and runs the same grader.

    python bench/grade_prompt_text.py --mode fl2va --length 345 example.txt

**A prompt is conformant AT A DURATION**, which is why every heading names one.
`S.SS` in Part One and any mid-shot time resolve against the snapped length,
and the action and dialogue are written to fill it, so the same text is correct
at 345 frames and wrong at 192 (the noodle bar, a 107-frame prompt, was
rendered at 345 for three days in September 2026 and its improvised tail was
read as an attention artifact). Pass
`--length` to grade at the duration the example is written for.

**For ref2va, pass `--like` a graph that wires the references the prompt
declares.** Reference labels are graded against the donor's sockets, so 10.5
reports two FAILs against a one-picture donor and none against
`h3_ref_image_audio_api`, which wires the two pictures and one audio clip it
names. The tool says so when every failure is of that shape.

### 10.1 T2VA — 243 frames, 10.125 s, two shots

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames a night ferry deck across the ten-second take, wet steel railing in the foreground and harbour lights smeared behind. A dock worker in her forties with a low, level alto (S1) leans on the rail, on-screen, unhurried delivery, and says: <d>[English] Last crossing until Thursday.</d> Her lips close and her jaw stops moving as she pushes back from the rail. A younger man in a canvas jacket stands two paces behind her and produces no vocal sound. The camera trucks right with small amplitude at slow speed, carrying the harbour lights across the frame. [Shot 2] The shot cuts to a close shot of the man's hands folding a paper timetable against his knee. The man, on-screen, mid-twenties, with a dry, slightly hoarse tenor (S2), looks up and answers: <d>[English] Then we wait it out here.</d> His lips settle closed and the paper flattens under his thumb.

overall_soundscape: Diesel engine rumble carries under a steady wash of water against the hull. Wind pulls at loose canvas, boots scuff on wet steel, and a mooring chain knocks twice against the deck plate.

non_diegetic_music: A single sustained low synth tone at a slow tempo, joined by a soft brushed-snare pulse that rises briefly and drops away before the final frame.
```

### 10.2 I2VA — five examples

#### 10.2.1 — 192 frames, 8.000 s, one shot

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the seated man shown in <Picture 1> remains at the workshop bench across the eight-second take, preserving his appearance, the grey work shirt, the bench layout, and the arrangement of tools behind him. The camera pushes in with small amplitude at slow speed as he turns the small brass mechanism a quarter turn under the lamp. Sawdust lifts in the lamp beam. The man, on-screen, sixties, with a quiet, gravelled baritone (S1), keeps his eyes on his hands and says: <d>[English] It was never the spring. It was the seat.</d> His lips close and his jaw stops moving, and he sets the mechanism down on the felt pad.

overall_soundscape: Low workshop room tone continues throughout under the hum of a bench lamp. Metal ticks against metal, felt brushes across wood, and a single drawer rolls shut near the end.

non_diegetic_music: N/A
```

#### 10.2.2 — 345 frames, 14.375 s

**No dialogue, one shot, camera opening out.** The picture's framing is held by the opening clause, and the only non-speaker on screen is given `produces no vocal sound`.

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, holding the exact framing, lighting, wardrobe and composition established in <Picture 1>, a bench joiner in a canvas apron stands over a half-planed board in a timber workshop, low sun coming through sawdust in the air behind her. She sets her weight, draws the plane the length of the board, and a single curled shaving lifts and falls to the floor. She produces no vocal sound. The camera pulls out with small amplitude at slow speed, opening the frame to the racked chisels along the back wall.

overall_soundscape: The long dry rasp of a hand plane on softwood repeats at an unhurried pace, each pass ending in the light tick of a shaving dropping to boards. Room tone is close and wooden, with a faint hum from a strip light overhead.

non_diegetic_music: N/A
```

#### 10.2.3 — 345 frames, 14.375 s

**Two shots, one speaking turn, dialogue in the second.** Neither header carries a time (section 3.1).

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, holding the exact framing, lighting, wardrobe and composition established in <Picture 1>, a woman in a grey linen shirt stands among staged seedling trays in a glasshouse, flat overcast light falling through the panes above her. She lifts one tray to eye level and turns it slowly. The camera pushes in with small amplitude at slow speed toward her hands. [Shot 2] The camera cuts to a close shot of the tray against her chest. The woman, on-screen, with a warm unhurried mezzo (S1), looks up and says: <d>[English] These two came up early.</d> Her lips close and her jaw stops moving as she sets the tray down on the bench.

overall_soundscape: A steady wash of rain on glass carries throughout, with the hollow knock of a plastic tray set on a wooden bench and the faint drip of condensation running down a pane.

non_diegetic_music: N/A
```

#### 10.2.4 — 345 frames, 14.375 s

**Documentary register, no dialogue, `non_diegetic_music` in use.** A tilt, which is one of the vocabulary's less-used types.

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, documentary, holding the exact framing, lighting, wardrobe and composition established in <Picture 1>, a fisherman in oilskins coils a wet mooring line on a harbour wall at first light, the hull of a small trawler filling the background. He works the coil twice around his forearm, drops it over a bollard, and straightens. He produces no vocal sound. The camera tilts up with small amplitude at slow speed from his hands to the masts behind him.

overall_soundscape: Water slaps against stone in an irregular rhythm under a constant low wind. Wet rope drags across concrete, a bollard takes the line with a dull knock, and gulls call intermittently at a distance.

non_diegetic_music: A sparse sustained string chord at a slow tempo, swelling gently as the frame opens upward and holding without resolution.
```

#### 10.2.5 — 345 frames, 14.375 s

**One shot, one speaking turn, mouth closed on the line.** Shows the identity-and-voice clause sitting immediately before the `<d>` block.

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, holding the exact framing, lighting, wardrobe and composition established in <Picture 1>, an archivist in a charcoal cardigan stands at a reading-room table under a green shaded lamp, a shallow box of loose photographs open in front of her. She lifts one print, angles it toward the lamp, and studies it. The archivist, on-screen, with a quiet precise alto (S1), says: <d>[English] This one was never catalogued.</d> Her lips settle closed and she lays the print face up on the table. The camera trucks left with small amplitude at slow speed along the edge of the table.

overall_soundscape: Deep room tone in a high-ceilinged reading room with a long soft reverb tail. Stiff photographic paper flexes and settles, and a chair creaks once somewhere off to the side.

non_diegetic_music: N/A
```

### 10.3 FL2VA — five examples

#### 10.3.1 — 192 frames, 8.000 s, one shot, no bracketed labels

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a kitchen table in flat morning light, beginning in the exact position and framing established by Picture 1 with a folded paper map lying closed beside a cooling mug. The camera holds a static shot through the entire eight-second duration. Two hands enter from the right, press the near edge of the map flat, and draw the first fold open; the paper lifts, creases release one at a time, and the printed coastline widens across the tabletop. The mug is nudged aside as the second fold opens, its shadow shortening as the sheet spreads over it. The map settles fully open, the hands withdraw to the frame edge, and the sheet, mug position and composition come to rest exactly as established by Picture 2 at the end of the shot.

overall_soundscape: Quiet kitchen room tone continues throughout with a faint refrigerator hum. Stiff paper crackles as each fold releases, ceramic scrapes briefly across wood, and the sheet settles with a soft rustle.

non_diegetic_music: Two alternating piano notes at a slow tempo, joined by a low sustained string that fades as the paper stops moving.
```

#### 10.3.2 — 345 frames, 14.375 s

**One shot, no dialogue.** The convergence is described as observable events -- strokes shortening, a hand taking the lane rope -- rather than as a transition into the picture.

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, one continuous shot in a tiled municipal pool hall, hard overhead light breaking on the water. A swimmer in a black cap pushes off the near wall and crosses the frame left to right in four unhurried strokes, the wake spreading behind her and flattening. She produces no vocal sound. The camera trucks right with small amplitude at slow speed, holding her shoulders in frame as she moves. Her stroke shortens, she reaches the far lane rope and takes hold of it, and her position, the settling water, the lighting and the camera's angle and framing converge on Picture 2 at the end.

overall_soundscape: Water breaks and closes over a steady four-count stroke rhythm, with a long hard reverb tail bouncing off tile throughout. A ventilation hum sits low under everything, and the lane rope rattles once as it is taken.

non_diegetic_music: N/A
```

#### 10.3.3 — 345 frames, 14.375 s

**Two shots, so Part One names `Shot 2`.** This is the case that no shipped graph exercises; see the note under section 13.

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide shot holds a chalkboard wall in an empty lecture room, late afternoon light raking across it from tall windows. A lecturer in a rolled-sleeve shirt works left to right, filling the board with diagrams, his back mostly to camera. He produces no vocal sound. The camera pans right with small amplitude at slow speed, following the writing as it advances. [Shot 2] The camera cuts to a medium shot from the far side of the room, the filled board now behind him. He sets the chalk in the tray, steps back once to take in the whole board, and folds his arms. His stance, the finished board, the raking light and the camera's angle and framing converge on Picture 2 at the end.

overall_soundscape: Chalk taps and drags against slate in short irregular bursts, each stroke ending with a dry click. The room carries a long empty reverb, with distant corridor footsteps passing once and fading.

non_diegetic_music: A single piano figure at a slow tempo, thinning to one sustained note as the writing stops.
```

#### 10.3.4 — 345 frames, 14.375 s

**Animation, one shot.** A style anchor in the opening clause, and a non-human subject given `produces no vocal sound`.

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Animation, glossy 3D CG, one continuous shot on a kitchen counter at night, a single pendant lamp throwing a warm pool of light onto the worktop. A short round robot with a dented copper shell and one oversized lens rolls in from frame left, stops at a spilled bag of flour, and begins sweeping it into a pile with a flat paddle arm. It produces no vocal sound. The camera pushes in with small amplitude at slow speed toward the worktop. The pile grows compact under the paddle, the robot lowers its lens toward it, and its position, the swept surface, the lamp's pool of light and the camera's angle and framing converge on Picture 2 at the end.

overall_soundscape: A fine granular sweep of powder across a hard worktop repeats in short strokes, with small servo whirs starting and stopping between them. The kitchen is otherwise quiet, with a refrigerator hum holding underneath.

non_diegetic_music: Light pizzicato strings at a moderate tempo, playful and even, settling to a single held note as the sweeping stops.
```

#### 10.3.5 — 345 frames, 14.375 s

**Two shots with one speaking turn.** Dialogue finishes in the first shot so the final frame can be still and closed-mouthed.

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames a guard in a navy uniform at a museum door, the gallery beyond her lit low for closing. She checks a wall clock, then walks the length of the doorway and puts her hand on the frame. The guard, on-screen, with a level unhurried contralto (S1), says: <d>[English] Two minutes, then we lock up.</d> Her lips close and her jaw stops moving as she turns back to the gallery. The camera trucks left with small amplitude at slow speed alongside her. [Shot 2] The camera cuts to a wide shot of the gallery from behind her, the far lights already out. She reaches the last switch, holds still a moment, and lowers her hand. Her position, the darkened gallery, the remaining doorway light and the camera's angle and framing converge on Picture 2 at the end.

overall_soundscape: Hard-soled footsteps carry across a stone gallery floor with a long cold reverb. Switches throw with a heavy mechanical clack, and the room tone drops noticeably as each bank of lights goes out.

non_diegetic_music: N/A
```

### 10.4 L2VA — five examples

#### 10.4.1 — 158 frames, 6.583 s, one shot

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 6.58-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a close shot opens on a school corridor at dusk, a girl in a green raincoat still walking with her back half to the camera and a closed locker door behind her, a plausible earlier state of the arrangement in <Picture 1>. The camera pulls out with small amplitude at slow speed as she slows, shifts the bag strap across her chest, and turns toward the lockers. The girl, on-screen, about twelve, with a light, slightly breathy voice (S1), says: <d>[English] I left it in here.</d> Her lips close and her jaw stops moving well before the end. She reaches up, presses the locker latch, and lets the door swing to the angle shown; her shoulders drop, her chin lifts, and her hand, the door angle, the corridor lighting and the exact final composition settle into <Picture 1>, mouth closed and still, as the shot ends.

overall_soundscape: Empty corridor reverb carries a distant door closing twice. Rubber soles squeak on polished floor, a bag strap slides across fabric, and a metal latch clicks once near the end.

non_diegetic_music: A sparse celeste figure at a slow tempo over one sustained low string, thinning to a single held note at the close.
```

#### 10.4.2 — 345 frames, 14.375 s

**One shot, no dialogue.** An inferred opening that arrives at the supplied last frame only at the final frame.

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, one continuous shot in a narrow record shop, afternoon light coming in through a window at the front. A man in a corduroy jacket works along a rack of sleeves, pulling one part way out, considering it, and pushing it back. He produces no vocal sound. The camera trucks right with small amplitude at slow speed, staying level with his hands as he moves down the rack. He slows, draws one sleeve fully out, and turns it to face him, and his stance, the held sleeve, the window light and the camera's angle and framing converge on the closing composition, reaching it only at the final frame.

overall_soundscape: Stiff cardboard sleeves slide and knock against each other in an irregular rhythm. The shop carries a close dry room tone, with muffled traffic through glass and the occasional creak of a floorboard.

non_diegetic_music: N/A
```

#### 10.4.3 — 345 frames, 14.375 s

**Two shots, so Part One names `[Shot 2]`.** Note the brackets: L2VA brackets both labels where FL2VA brackets neither.

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot 2]) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide shot holds a rooftop at dusk, city haze behind and a folded deck chair by the parapet. A woman in a long cardigan steps out through a stairwell door and crosses toward the parapet, unhurried. She produces no vocal sound. The camera pans right with small amplitude at slow speed to follow her across the roof. [Shot 2] The camera cuts to a medium shot from beside the parapet as she arrives. She unfolds the deck chair, sets it square to the view, and lowers herself into it. Her seated position, the opened chair, the fading dusk light and the camera's angle and framing converge on the closing composition, reaching it only at the final frame.

overall_soundscape: A steady rooftop wind carries throughout with occasional gusts. A metal door swings shut once behind her, the chair frame clicks as it opens, and distant traffic hums many floors below.

non_diegetic_music: A slow warm synthesizer pad, building gradually in dynamics and holding as the movement comes to rest.
```

#### 10.4.4 — 345 frames, 14.375 s

**Animation, one shot.** Shows that `<Picture 1>` is the FINAL frame and is not part of `[Shot 1]`'s opening composition.

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Animation, hand-drawn, one continuous shot in a cluttered attic study lit by a single desk lamp. A tall thin fox in a knitted waistcoat searches a stack of papers, lifting sheets aside two at a time and setting them down in a growing pile. He produces no vocal sound. The camera pushes in with small amplitude at slow speed toward the desk. His hands slow, he draws one folded sheet from near the bottom of the stack and opens it flat under the lamp, and his posture, the opened sheet, the lamp's pool of light and the camera's angle and framing converge on the closing composition, reaching it only at the final frame.

overall_soundscape: Paper rustles and slides in short overlapping strokes throughout, with the soft thump of sheets settling into a pile. The attic is close and quiet, with rain faint on a roof overhead.

non_diegetic_music: A solo clarinet line at a slow tempo, curious and unhurried, thinning to a single sustained note as the searching stops.
```

#### 10.4.5 — 345 frames, 14.375 s

**Two shots with one speaking turn.** The line lands early, leaving the close silent.

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot 2]) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, documentary, a medium shot frames a farrier in a leather apron at the open side of a stable yard, cold morning light flattening the scene. He works a rasp along the edge of a hoof held between his knees, steady and repetitive. The farrier, on-screen, with a low weathered baritone (S1), says: <d>[English] Nearly there, stand easy.</d> His lips close and his jaw stops moving as he returns to the rasp. The camera holds a static shot. [Shot 2] The camera cuts to a wide shot of the yard as he finishes. He lowers the hoof, straightens up, and rests one hand on the horse's shoulder. His standing position, the settled horse, the flat morning light and the camera's angle and framing converge on the closing composition, reaching it only at the final frame.

overall_soundscape: A rasp draws across horn in long even strokes with a dry grain to each pass. Hooves shift on wet cobbles, a bucket handle rings once, and the yard carries an open outdoor ambience with birdsong at a distance.

non_diegetic_music: N/A
```

### 10.5 ref2va — 243 frames, 10.125 s, two shots, two images and one audio clip

Reference chain in connection order: image (the woman), image (the workshop),
audio (a voice clip). Labels therefore resolve to `<Picture 1>`, `<Picture 2>`,
`<Audio 1>`. Both images define identity and environment only, so per ref §2.2
neither gets a standalone entry and neither is cited in `detailed_description`.

```text
subject_definitions:
<Subject 1> is the woman in <Picture 1>, in her thirties, with cropped black hair, a tan canvas apron over a grey long-sleeved shirt, and a thin leather cord at her wrist.
<Subject 2> is the bicycle-repair workshop in <Picture 2>, with a whitewashed brick wall, a pegboard of hanging tools, a bare bulb over the bench, and two frames on a floor stand.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1), carrying a spoken English vocal layer at a low, even pitch.

summary:
[reference generation + audio reference] The target video shows <Subject 1> truing a wheel in <Subject 2> while a customer waits at the doorway. The two-shot exchange uses <Audio 1> as the voice-timbre reference for <Subject 1> and ends with the wheel spinning clean.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - the cropped black hair, tan canvas apron, grey long-sleeved shirt, and leather wrist cord are retained.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the whitewashed brick wall, pegboard of hanging tools, bare bulb, and floor stand are retained.
<Audio 1>: reference - its vocal timbre and even delivery guide the speech of <Subject 1> without copying the original signal.

detailed_description:
The target video is in a warm, naturalistic documentary style with a single practical light source and a slightly muted colour palette.
[Shot 1] A medium shot establishes <Subject 2>, the bicycle-repair workshop, its whitewashed brick wall and pegboard of hanging tools lit by one bare bulb over the bench. <Subject 1> (S1), the woman with cropped black hair and a tan canvas apron over a grey long-sleeved shirt, stands at the floor stand on the right of frame with a rear wheel clamped in the truing jig, a spoke key held between two fingers of her right hand. She rotates the wheel a half turn, watches the rim pass the indicator, and stops it with the flat of her palm. The camera pushes in with small amplitude at slow speed toward the jig as the rim comes to rest. A man in a wet cycling jacket stands just inside the doorway at the left edge of frame, helmet under one arm, and produces no vocal sound. Using the low, even timbre referenced from <Audio 1>, <Subject 1> (S1) says without looking up, <d>[English] Two spokes are loose.</d> Her lips close and her jaw stops moving as she fits the spoke key onto a nipple and turns it a quarter turn.
[Shot 2] The shot cuts to a close shot over her shoulder, the rim filling the lower half of the frame and the pegboard soft behind it. Her thumb steadies the rim while the spoke key turns twice more, and the leather cord at her wrist slides against the apron edge. The camera holds a static shot as she releases the rim and sets it spinning. The man in the wet cycling jacket, on-screen at the frame edge, mid-forties, with a light, hesitant tenor (S2), steps forward and asks, <d>[English] Can I still ride it home?</d> His lips close and he shifts the helmet to his other arm. <Subject 1> (S1) watches one full rotation, then answers in the same low, even timbre referenced from <Audio 1>, <d>[English] Yes. Slowly.</d> Her lips settle closed and her jaw stops moving while the wheel keeps turning, the rim passing the indicator without touching it, and the bare bulb throws a moving band of light across the pegboard behind her.

overall_soundscape:
Quiet workshop room tone and a faint street hum continue underneath throughout. A spoke key ticks against metal in short bursts, a wheel rim hums as it spins down, and wet fabric creaks as the waiting man shifts his weight.

non_diegetic_music:
N/A
```

`detailed_description` here runs about 380 words, inside ref §5.2's 350-500 for a
generation task. That length is the point of the example: a two-sentence ref2va
body is the most common defect in this repo's shipped prompts, and a short
example would teach it.

---

