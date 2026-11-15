# Sound Cues and Timed Mask "Blooms" (Plain-English Guide)

Last updated: 2026-11-15.

**Quick answer:** If you saw the label **`bloom_soundcues`** in an experiment folder (`$H3_OUTPUT_DIR/Video/nov15_lab/exp2_av_bloom_soundcues/`), it is **not** a single node or feature. It was shorthand for pairing **two separate timing tools** in one test:

1. **Sound Cues** ([`MiniMaxH3SoundCues`](../../../sound_cues.py)) — invisible drumbeats on the timeline that tell the video+audio model **when** an impact, hop, footstep, or snap happens.
2. **Timed Mask "Blooms"** ([`MiniMaxH3EditGuider`](../../../edit_guidance.py) with an animated black-and-white mask video) — a spotlight that stays closed during calm parts of the clip and opens ("blooms") over one area of the screen only during the frames you pick.

Read [`which_tool_and_what_to_call_it.md`](which_tool_and_what_to_call_it.md) first if you want the map of Control Vectors, Edit Guidance, and the Word Readout (`J-Lens`). This page explains how **time- and region-locked controls** add frame-level timing on top of them.

---

## 1. Why two extra tools exist alongside text prompts

A normal prompt is like a movie poster summary for the whole clip: it describes who is in the room, what happens, and what the room sounds like, **without a frame-by-frame stopwatch or a map of the screen**.

| Tool | Plain-English job | Gives the model... | Cost vs. base render |
|---|---|---|---|
| **Text prompt alone** | Sets the overall scene, look, action, and audio mood | Words (no frame numbers, no screen coordinates) | `1x` |
| **Sound Cues** ([`MiniMaxH3SoundCues`](../../../sound_cues.py)) | Pins physical hits, jumps, or steps to exact timestamps | Temporary in-memory thuds/snaps at chosen frames, removed partway through sampling | `1x` (or `~1.5x` with Edit Guidance) |
| **Timed Mask "Bloom"** (`mask` on [`MiniMaxH3EditGuider`](../../../edit_guidance.py)) | Turns a prompt edit ON only over a chosen area and chosen frames | A frame-by-frame mask video (`0` = keep original take, `1` = steer toward edited prompt) | `~1.5x–2x` (runs two passes on guided steps) |

Put simply:
- **Sound Cues** answer **"At which second should the strike land?"**
- **Timed Mask Blooms** answer **"Where on screen, and for which frames, should the visual fireworks happen?"**

---

## 2. Part 1: How Sound Cues work (without audio engineering jargon)

### The problem
If you write *"the tortoise jumps at 1.0s, 2.25s, and 4.0s"* inside a text prompt, MiniMax H3 sees English words, not a clock. Because its video+audio generator prefers natural, evenly spaced motion, jumps drift to whatever rhythm the model chooses on its own.

### What `MiniMaxH3SoundCues` does
MiniMax H3 generates picture and sound **together in one timeline**, so its eyes and ears constantly listen to each other while a shot forms. [`MiniMaxH3SoundCues`](../../../sound_cues.py) uses that shared ear to steer body motion:

1. **You write a tiny beat sheet in frames or seconds (no `.wav` files needed).**  
   At `24` frames per second, you type one line per beat:
   ```text
   24: hop
   54: snap
   60: slam
   ```
   (Or in seconds, like `1.0s: slam` and `2.5s: impact`.) Built-in presets are `hop` / `thud`, `snap` / `yank`, `slam` / `impact`, `yank_slam`, `step`, and `click`.
2. **It builds tiny temporary sound bumps in memory.**  
   Instead of saving files on disk, the node creates short percussive clicks/thuds at those timestamps, mixes a soft `-28 dBFS` low room hum underneath (so H3 does not mistake empty space between cues for a muted room), nudges timestamps onto `3`-frame (`1/8`-second) marks where video (`24 fps`) and audio (`40 Hz`) line up cleanly, and merges touching cues so two nearby hits never collide.
3. **You take the training wheels off before the render finishes (`0.33` release).**  
   Wait—if you feed synthetic drum bumps into the generator, won't your finished video have fake beep/thud noises?  
   Not if you wire [`MiniMaxH3ReleaseReferences`](../../../control_vector_nodes.py) (`audio_end_percent = 0.33` on a `6`-step Pika render, or `0.60` when paired with `MiniMaxH3EditGuider(streams="video_only")`):
   - **Steps 0–1 (rough layout):** H3 hears the cue beats on the timeline and lines up the body wind-up, footfall, or hammer strike onto those beats.
   - **Steps 2–5 (fine detail & sound):** `MiniMaxH3ReleaseReferences` hides the synthetic cues. H3 already committed to the motion timing, so it fills in natural street footsteps, wind, or anvil ringing instead of synthetic thuds.

> **Tip from testing:** Put each sound cue on the **landing or impact frame**, not the start of a jump. When given a landing cue, H3 naturally starts lifting off several frames earlier so the foot or hammer *arrives* on the beat.

---

## 3. Part 2: How Timed Mask "Blooms" work

### The problem
With [`MiniMaxH3EditGuider`](../../../edit_guidance.py), you compare a **base prompt** (`original`) against an **edited prompt** (`edited`). Without a mask, that push affects the **whole picture for the whole clip**—so asking for *"blazing blue-white lightning arcs erupting explosively"* can wash the whole room in blue strobe light even before anything happens.

### Three ways to mask `MiniMaxH3EditGuider`
You can hand any ComfyUI `MASK` into `MiniMaxH3EditGuider.mask`, where **`0` (black) stays on the base prompt** and **`1` (white) steers toward the edited prompt**:

1. **A still black-and-white shape** (`1` frame, spatial):  
   Locks one part of the screen to your base render (or applies an edit only inside a painted area) for the entire clip—for example, keeping a blizzard off a walking tortoise ([`bench/results/2026-11-14_edit_guider_mask_ab.md`](../../../bench/results/2026-11-14_edit_guider_mask_ab.md)).
2. **A full-screen time envelope** (`[frames, 1, 1]`):  
   Turns an edit up and down across time for the entire frame and soundtrack together.
3. **An animated mask "bloom"** (`[frames, height, width]` black-and-white video):  
   Combines **where** and **when** in one mask. Frames can start completely black (`0`), swell open like a soft spotlight over one prop during Beat 1 (`0 -> 1 -> 0`), close back to `0` while the actor resets, then swell open wider during Beat 2 before closing again.  
   *(Important detail: whenever a mask has 2D screen dimensions `H > 1` or `W > 1`, the mask controls **which video regions change**, while soundtrack steering follows `streams` and `audio_scale` smoothly across the clip.)*

### Pick an audio+video aligned clip length
Whenever you use an animated multi-frame mask (`[frames, height, width]`) or audio-coupled render, keep the clip length on H3's shared audio-video grid (`39`, `90`, `141`, `192`, `243`, `294`, or `345` frames at `24 fps`; see [`h3_rules.py`](../../../h3_rules.py)). That keeps video frames, audio tokens, and mask frames ending on the exact same beat.

---

## 4. What we actually proved (and what we didn't)

Two separate tests proved two different parts of this story. Keeping them apart makes it clear what each tool is actually doing.

### Proof A (`2026-11-12`): Sound Cues alone lock timing without synthetic audio bleed
In the `124`-frame tortoise crosswalk runs ([`2026-11-12_sound_cues_dpad_and_033_release_law.md`](../research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md)):

- **Without Sound Cues:** Prompt text asking for three unevenly spaced hops missed the requested beat timestamps and fell back to an even forward pace.
- **With `MiniMaxH3SoundCues` (`24: hop`, `54: snap`, `57: slam`, `96: hop`) + `0.33` release:** The render hit `3/3` requested hops and the mid-clip red-leash snap on cue at normal `1x` speed (`6` steps), while the finished soundtrack stayed natural street ambience (near-zero correlation with the synthetic helper waveforms).
- **Combined with a 2D cutout guide ([`MiniMaxH3AppendSpriteTrajectory`](../../../sprite_trajectory.py)):** Releasing synthetic references at `0.33` let a tortoise walk forward, jump on time, and walk **backward while still facing right** with real alternating leg strides instead of gliding like a stiff paper cutout.

### Proof B (`2026-11-15`): Animated mask "blooms" fire localized visual FX mid-shot and leave the set alone
In the `90`-frame (`3.75s` at `24 fps`, `1344x768`, `seed = 42`) blacksmith forge side-by-side (`$H3_OUTPUT_DIR/Video/nov15_lab/exp2_av_bloom_soundcues/`):

- **What was wired:** **Both** the `Base` clip and the `Bloom` clip used the exact same `90`-frame prompt and the same `MiniMaxH3SoundCues` beats (`24: slam`, `60: impact`). The **only** addition in `Bloom` was `MiniMaxH3EditGuider` (`scale = 2.2`, `first_step = 1`, `last_step = 3`, `streams = "independent"`, `audio_scale = 1.2`) swapping the calm impact description for *"a dazzling blinding electric cyan-violet plasma shockwave ring and blazing blue-white lightning arcs erupt explosively outward from the anvil core"*, gated by a soft circular `[90, H, W]` mask centered over the anvil that only opened in two windows (`frames 24–43` and `frames 60–83`).
- **What actually happened on screen and in audio (`exp2_base_vs_bloom_stacked.1fps.00000-00089.mp4`):**
  - **`frames 0–23` (mask closed):** Identical blacksmith, cave walls, hanging lanterns, tool racks, and opening anvil strike (`frame 6`).
  - **`frames 24–43` (first bloom opens over center):** `Bloom` lifts the hammer noticeably higher overhead (`frames 26–35`) into a heavy second strike (`frame 40`, loud anvil ring peaking around `frame 48`). Meanwhile, unmasked cave walls and lanterns stay locked (`~7.1` out of `255` average pixel difference).
  - **`frames 44–59` (mask closed again):** Both clips return to calm smithing glow; `Base` lowers its hammer after `frame 45` and stops striking.
  - **`frames 60–83` (second wider bloom opens over center):** While `Base` rests the hammer calmly on the anvil, `Bloom` charges a glowing violet plasma core on the anvil (`frames 64–74`) as the blacksmith raises the hammer overhead, brings it crashing down (`frame 75`), and erupts into a bright **electric cyan-white lightning burst (`frames 76–79`)** plus a loud third anvil ring (`frame 82`) that settles back into ordinary orange sparks (`frames 80–89`) once the mask closes.
  - **Spatial stability:** Across the whole clip, pixels permanently outside the circular bloom zone moved just `7.10 / 255` during quiet stretches and `7.05 / 255` during active explosions—compared to `12.85` average (`22.2` peak at `frame 26`) inside the active center bloom.

### Honest caveats (what the forge test did *not* prove, and three practical gotchas)
1. **The forge clip tested the animated bloom mask—not Sound Cues alone.** Because both `Base` and `Bloom` already had `MiniMaxH3SoundCues` wired (`24: slam`, `60: impact`), the forge A/B isolates what adding a timed `[90, H, W]` Edit Guidance mask accomplishes. Sound Cues vs. plain text was settled separately by the `2026-11-12` tortoise runs.
2. **Expect ~0.5 seconds (`12–15` frames) of physical wind-up.** Opening a visual mask at `frame 60` lets the edited action *start* around `frame 60`: the blacksmith still has to lift his arm (`frames 62–74`) before the hammer lands and triggers the cyan lightning flash (`frames 75–79`). If you want the peak flash to land *right at* `frame 60`, start opening the bloom mask `~12–15` frames earlier (`~frame 45`).
3. **Why the review strip shows `inMAE=0.0` on quiet (and tail-flash) frames.** The per-frame measurement script averages pixel change only where the mask exceeds `0.5`. When the mask is closed (`0.0`) or already tapering down (`frames 75+`), no pixel crosses `0.5`, so `inMAE` prints `0.0` by rule—even while the lightning flash finishes decaying over `frames 76–85`. Look at the background metric (`bg ~ 7.1`) for true off-mask stability.

---

## 5. Quick recipe: which setup should I pick?

| What you want | Use this | Starting workflow |
|---|---|---|
| **Hops, footsteps, or impacts hitting exact seconds** in a normal text-to-video clip | `MiniMaxH3SoundCues` + `MiniMaxH3ReleaseReferences` (`audio_end_percent = 0.33` on `BasicGuider`) | [`workflows/h3_text_to_video_pika_sound_guide_api.json`](../../../workflows/h3_text_to_video_pika_sound_guide_api.json) |
| **Keep a `<Picture 1>` character + control where they walk/turn/jump** + hit sound beats | `MiniMaxH3AppendSpriteTrajectory` + `MiniMaxH3SoundCues` + `MiniMaxH3ReleaseReferences` (`video_end_percent = 0.33`) | [`workflows/h3_image_ref_dpad_sound_pika_api.json`](../../../workflows/h3_image_ref_dpad_sound_pika_api.json) |
| **Localized visual burst or effect only on certain beats** without re-lighting the whole room | `MiniMaxH3EditGuider` (`streams = "independent"`, `audio_scale = 1.0..1.2`, `first_step = 1`, `last_step = 3`) + `[F, H, W]` mask video (+ optional `MiniMaxH3SoundCues` on `audio_end_percent = 0.60`) | Start from [`workflows/h3_text_to_video_pika_edit_guidance_api.json`](../../../workflows/h3_text_to_video_pika_edit_guidance_api.json) (wire `LoadVideo -> GetVideoComponents -> ImageToMask` into `MiniMaxH3EditGuider.mask`; pick a `39 / 90 / 141 / ...` length) |
| **Static region protection or swap** (keep a subject clean or edit one prop all clip) | `MiniMaxH3EditGuider.mask` with a single-frame mask (buffer protection masks with `GrowMask` + `ImageBlur` -> `InvertMask`) | See [`../edit_guidance.md`](../edit_guidance.md) §3 and [`bench/results/2026-11-14_edit_guider_mask_ab.md`](../../../bench/results/2026-11-14_edit_guider_mask_ab.md) |
