# Sound Cues, 2D Sprite Trajectories, and the `0.33` Reference-Release Rule (`2026-11-12`)

When a prompt asks MiniMax H3 for specific timing or off-prior motion—landing three hops on exact frames across a five-second shot, or stopping mid-crosswalk and walking backward while still facing forward—text conditioning alone rarely delivers all of the constraints at once. Text lives in a global sequence header with no frame coordinates, and H3's motion prior favors ordinary forward locomotion and evenly spaced beats.

Two timeline conditioning nodes and one selective release rule address this inside standard Pika `6`-step workflows without training LoRAs or rendering external `.wav` or `.mp4` helper files:

- **[`MiniMaxH3SoundCues`](../../../sound_cues.py)** synthesizes short `32 kHz` percussive islands from a text beat sheet (`24: hop`, `54: snap`, `57: slam`, `96: hop`) over a `-28 dBFS` low-frequency room floor and anchors them at exact video frames as `minimax_keyframes` `cond_audio` tokens.
- **[`MiniMaxH3AppendSpriteTrajectory`](../../../sprite_trajectory.py)** cuts the subject out of `<Picture 1>` in memory and renders a compact `384x224` 2D kinematic reference clip (`<Video 1>`) from plain-text `(x, y, face)` waypoints and parabolic `hop` arcs, decoupling travel direction from body orientation.
- **[`MiniMaxH3ReleaseReferences`](../../../control_vector_nodes.py)** drops those synthetic `<Video N>` and `cond_audio` tokens at **`0.33`** (`2` of `6` Pika steps, corresponding to unshifted flow-matching midpoint **`b = 0.50`**) while keeping `<Picture 1>` at **`end_percent = 1.0`** across all `6` steps so subject identity stays locked while H3 articulates legs, shadows, and natural street foley on steps `2..5`.

 Companion records:
- Raw experiment script, graphs, and per-arm metrics: `data/ref_signal_2026-11-12/`
- Chronological lab notebook (`exp1_sound` and `exp2_dpad` initial runs): [`docs/research/2026-11-12_guide_sound_islands_and_kinematic_dpad.md`](../2026-11-12_guide_sound_islands_and_kinematic_dpad.md)
- Node reference & wiki routers: [`docs/wiki/control_vectors/comfyui_nodes_and_controls.md`](../../wiki/control_vectors/comfyui_nodes_and_controls.md), [`docs/wiki/edit_guidance.md`](../../wiki/edit_guidance.md), [`docs/wiki/h3_uses.md`](../../wiki/h3_uses.md)

---

## 1. Here's an example

Suppose you have a single still image (`$H3_INPUT_DIR/ref_signal/tortoise_pic1.png`, wired as `<Picture 1>`) showing a grey tortoise with a lit white birthday candle standing upright on top of its shell, and you want a `124`-frame (`5.17s` at `24 fps`, `1344x768`) shot on an overcast city crosswalk where the tortoise:

1. Starts on the left-center stripes facing right (`f00`),
2. Walks forward to the right half of the road (`f00..f44`),
3. Leaps upward into the air and lands cleanly around `f51..f54` (`~2.15..2.25s`) with the candle still lit,
4. Pauses briefly (`f54..f66`), and
5. Walks **backward** from right to left (`f66..f112`) **while keeping its head and shell facing right**.

### What happens without timeline anchors
- **Prompt alone (`exp2_dpad` Arm 1, `unet_ref2va`, `seed = 104`):** The tortoise walks steadily right (`cx: 451 -> 888 px`), executes the mid-shot hop, and then stays near `cx = 865..888 px` through `f120` (`0 px` of backward travel). In H3's training prior, a right-facing quadruped continues right or stops.
- **Holding a 2D cutout video across all 6 steps (`exp2_dpad` Arm 3, `video_end_percent = 1.0`):** The tortoise follows the rightward walk, jumps, and reverses `-353 px` leftward while still facing right (`cx: 416 -> 843 -> 490 px`), keeping the candle lit—but its jump peaks `7` frames late (`f56` vs `f49`) and its legs stay frozen in `<Picture 1>`'s static standing pose across `f68..f92`, gliding backward like a rigid cutout.

### What happens in the shipped hybrid workflow
**[workflows/h3_image_ref_dpad_sound_pika_api.json](../../../workflows/h3_image_ref_dpad_sound_pika_api.json)** combines `<Picture 1>`, `MiniMaxH3AppendSpriteTrajectory` (`<Video 1>`), a touchdown `hop` cue at `f54` in `MiniMaxH3SoundCues`, `MiniMaxH3ReleaseReferences(end_percent=1.0, video_end_percent=0.33, audio_end_percent=0.60)`, and `MiniMaxH3EditGuider(streams="video_only", scale=1.6, last_step=3)`.

Measured on the matched seed (`Gate 2b`, `seed = 104`, `1344x768x124`):
- **Forward walk + timed jump:** Walks right (`cx: 392 -> 754 px`), lifts off on time, peaks at **`f51`** (`5` frames closer to the sprite's `f49` apex than the video-only run), and lands at `f54..f55`.
- **Facing-right backward walk with articulated legs:** Reverses **`-350 px`** from `cx = 816 px` (`f64`) back to `cx = 466 px` (`f112`) while keeping head and shell pointed `+x` (`head_dx = +100..+119 px`), and executes **three alternating backward leg-stride cycles** across `f72..f92` (`f72, f80, f88` swing vs `f76, f84, f92` plant) instead of sliding as a frozen cutout.
- **Identity and audio:** The upright birthday candle stays lit across `f000..f120`, and the final soundtrack has zero correlation with the synthetic `f54` cue (`r = -0.0196`) over a natural `-33.4 dBFS` street-ambience floor.

### Companion text-to-video example (timed beats without a reference image)
When you only need frame-accurate event timing in text-to-video (`unet_fl2va`, no `<Picture 1>` or `<Video 1>`), **[workflows/h3_text_to_video_pika_sound_guide_api.json](../../../workflows/h3_text_to_video_pika_sound_guide_api.json)** (or `MiniMaxH3SoundCues` with `24: hop`, `54: snap`, `57: slam`, `96: hop`) locks all `3/3` tortoise hops (`f24`, `f56..57`, `f96`) and the mid-clip red-leash snap while keeping natural street foley (`r = -0.1578` with `BasicGuider` + `audio_end_percent = 0.33` at `60.3s`, or `r = -0.0086` with `EditGuider(streams="video_only")` + `audio_end_percent = 0.60`).

---

## 2. Here's how you do it

### Recipe A — Timed sound events on a text-to-video graph (`1x` cost, 6 DiT passes)
Use this when your prompt already describes the scene and motion, and you want impacts, hops, footsteps, or snaps to land on specific frames without increasing render time:

1. Start from a standard `unet_fl2va` Pika `6`-step graph (`ManualSigmas` + `SamplerCustomAdvanced` + `BasicGuider`).
2. Insert **`MiniMaxH3SoundCues`** between `MiniMaxH3Conditioning` and `MiniMaxH3Preflight`:
   - Wire `positive` and `latent` from `MiniMaxH3Conditioning`, plus `audio_vae` from `VAELoader` (H3 Audio VAE).
   - Enter one cue per line in `cues` using frame numbers (`24 fps`) or seconds (`s`):
     ```text
     24: hop
     54: snap
     57: slam
     96: hop
     ```
   - Leave `snap_to_3f = True` and `floor_dbfs = -28.0`.
3. Insert **`MiniMaxH3ReleaseReferences`** on the `model` wire upstream of `BasicGuider`:
   - Set `end_percent = 1.0` and **`audio_end_percent = 0.33`** (releases synthetic `cond_audio` after step `1`, before step `2` runs).
   - If you prefer stronger early visual pull on subtle cues and accept `10` DiT passes instead of `6` (`~85s` vs `~55s`), use `MiniMaxH3EditGuider(scale=1.6, last_step=3, streams="video_only")` with `original` wired to `MiniMaxH3Conditioning.positive` (no sound cues) and `edited` wired to `MiniMaxH3Preflight` (with sound cues), setting `audio_end_percent = 0.60`.

### Recipe B — Subject still + 2D trajectory + timed sound cues (`unet_ref2va`)
Load **[workflows/h3_image_ref_dpad_sound_pika_api.json](../../../workflows/h3_image_ref_dpad_sound_pika_api.json)** (or build the LiteGraph UI workflow via `python workflows/build_workflows.py`):

1. **Load your subject image (`LoadImage` -> `MiniMaxH3AppendRefImage`):** This becomes `<Picture 1>`. If the subject sits on a clean or separable margin, no mask is required; if the background is cluttered or touches the image border, wire a `MASK` into `MiniMaxH3AppendSpriteTrajectory.mask`.
2. **Script the 2D path in `MiniMaxH3AppendSpriteTrajectory`:**
   - Wire `references` from `MiniMaxH3AppendRefImage` and `image` from `LoadImage`. This appends `<Video 1>` to the typed reference chain.
   - Write keyframes in normalized canvas coordinates (`x` in `0..1` left-to-right, `y` in `0..1` top-to-bottom foot contact, `face=right|left|auto`, optional `scale=` and `hop start..end: lift=...`):
     ```text
     0: x=0.31, y=0.69, face=right, scale=1.0
     44: x=0.63, y=0.69, face=right
     hop 44..54: lift=0.14
     54: x=0.69, y=0.69, face=right
     66: x=0.69, y=0.69, face=right
     112: x=0.33, y=0.69, face=right
     123: x=0.33, y=0.69, face=right
     ```
   - `source_facing` tells the node which way the subject already faces inside `<Picture 1>` (`right` or `left`), so `face=right` keeps that orientation even while `x` decreases from `0.69` to `0.33`.
   - Connect `preview_frames` (`IMAGE` slot `1`) to `PreviewImage` or `VHS_VideoCombine` if you want to inspect the `384x224` sprite plate before sampling.
3. **Anchor impact timing in `MiniMaxH3SoundCues`:**
   - Place the audio cue at the **landing / contact** frame (`54: hop` for `hop 44..54`), not the liftoff frame (`44`).
4. **Set selective release on `MiniMaxH3ReleaseReferences`:**
   - `end_percent = 1.0` (keeps `<Picture 1>` active on all `6` steps),
   - **`video_end_percent = 0.33`** (releases `<Video 1>` after `2` steps so H3 animates legs and contact pose naturally on steps `2..5`),
   - `audio_end_percent = 0.60` when paired with `MiniMaxH3EditGuider(streams="video_only", scale=1.6, last_step=3)` (or `0.33` when paired with `BasicGuider`).
5. **Write a matching 6-section `ref2va` prompt** (example in [`prompt_bank/ref2va_dpad_tortoise_sound_cues.txt`](../../../prompt_bank/ref2va_dpad_tortoise_sound_cues.txt)) defining `<Subject 1>` from `<Picture 1>` (`fully_preserved`) and `<Video 1>` as motion-only (`attribute_transfer`).

---

## 3. What each part in the workflow does

In [`workflows/h3_image_ref_dpad_sound_pika_api.json`](../../../workflows/h3_image_ref_dpad_sound_pika_api.json), each node owns one piece of the conditioning contract:

| Node in graph | Role | What it sends into H3 | Active steps (`6`-step Pika) |
|---|---|---|---|
| `LoadImage` + `MiniMaxH3AppendRefImage` (`34 -> 50`) | Identity anchor (`<Picture 1>`) | Full-detail Qwen3-VL vision tokens + 3D VAE `ref_img` latents of the subject | Steps `0..5` (`end_percent = 1.0`, all `6` steps) |
| `MiniMaxH3AppendSpriteTrajectory` (`51`) | Coarse spatial & facing guide (`<Video 1>`) | `384x224` cutout trajectory clip (`+3,108` `ref_img` DiT rows + `504` Qwen tokens at `allow_upscale=False`, no audio) | Steps `0..1` (`video_end_percent = 0.33`, dropped at step `2`) |
| `MiniMaxH3ReferenceConditioning` (`5`) | Joint text + reference compiler | Encodes the 6-section `ref2va` prompt alongside `<Picture 1>` and `<Video 1>`, plus the empty `1344x768x124` AV target latent | Text active steps `0..5` |
| `MiniMaxH3SoundCues` (`70`) | Frame-accurate acoustic beat sheet | Synthesizes `32 kHz` stereo cue islands + `-28 dBFS` rumble floor, Audio-VAE encodes them once, and attaches `minimax_keyframes` (`cond_audio`) | Steps `0..3` (`audio_end_percent = 0.60` under `EditGuider video_only`, or `0..1` at `0.33` under `BasicGuider`) |
| `MiniMaxH3ReleaseReferences` (`135`) | Per-modality temporal gate | Strips `<Video 1>` at `0.33` and `cond_audio` at `0.60` (or `0.33`) from `PackedLayout` mid-denoise while keeping `<Picture 1>` through `1.0` | Wraps `unet_function` on every step |
| `MiniMaxH3EditGuider` (`9`) | `1x`-Qwen video-only contrast guider | Runs `orig + 1.6 * (edited - orig)` on the **video** stream across steps `0..2` (`last_step = 3`) while feeding `orig` (no synthetic audio) to the **audio** stream | Steps `0..2` (`2` passes/step), then `1` pass/step on `3..5` (`10` passes total) |

---

## 4. Why it works

Joint video-and-audio diffusion in MiniMax H3 processes video tokens, audio tokens, reference latents, and text tokens inside a single shared 3D/1D Rotary Position Embedding (`RoPE`) timeline:

1. **Timeline tokens have exact clock and screen coordinates; text does not.**
   In H3, Qwen3-VL text tokens sit in a positionless prefix block before frame `0`. When a prompt asks for three unevenly spaced hops (`1.0s`, `2.25s`, `4.0s`) or a mid-shot direction reversal, the DiT has to translate English words into `40 Hz` / `24 fps` RoPE coordinates across `37,296` video tokens. By contrast, a `cond_audio` island anchored at `frame_idx = 54` sits at exact RoPE coordinate `t = 90.0`—the identical timestamp as video frame `54`—and `<Video 1>` places the subject's pixels on the left or right half of the canvas at every frame. Cross-modal self-attention locks onto those coordinates during the first two denoising steps (`steps 0..1`), when coarse layout and motion trajectories form.

2. **Half of unshifted flow time happens in the first two Pika steps.**
   Because H3 uses a heavy time shift (`shift = 12.0` on video, `3.0` on audio), Pika's `6` steps are front-loaded near `sigma = 1.0`. The first `2` steps (`0.33` of the step count) already advance the underlying unshifted flow-matching clock `b` from `1.00` to **`0.50`**—halfway through flow time. By `b = 0.50`, coarse body trajectory (`cx(t)`), facing polarity (`+x` vs `-x`), and jump timing are already committed in the noisy latent `x_t`.

3. **Releasing synthetic scaffolding at `0.33` (`b = 0.50`) lets H3 finish the physics and foley.**
   Synthetic helpers carry unwanted fine-detail priors if held too long: a 2D affine cutout has frozen legs and a flat grey background, and a procedural `.wav` bursts with synthetic noise/sine transients. Dropping `<Video 1>` and `cond_audio` at `0.33` removes their tokens from DiT self-attention for the entire second half of unshifted flow time (`b: 0.50 -> 0.00`, steps `2..5`), while `<Picture 1>` stays at `end_percent = 1.0` to preserve shell pattern, head texture, and the lit birthday candle. Left with a trajectory that is already moving leftward while facing right—and no frozen cutout forcing static legs—H3 resolves that velocity field the only physically consistent way its weights know how: by articulating alternating backward leg strides and synthesizing natural street ambience.

---

## 5. What doesn't work, what's still hacky, and what could be better

Every failure mode below was observed in the `2026-11-12` runs (`data/ref_signal_2026-11-12/`) and shapes the current defaults and caveats:

1. **Holding synthetic `.wav` islands to `1.0` bleeds clicks and mutes room tone if the `.wav` has digital-zero tails.**
   - In `exp1_sound` Arm 2 (`audio_end_percent = 1.0`, zero-floor `.wav` islands), synthetic bursts leaked into the output audio (`r = +0.16..+0.23`), and inter-event street ambience dropped by **`~20 dB`** (`-52.9 dBFS` vs `-33.0 dBFS` baseline) because `-94 dBFS` digital-zero tails taught the Audio DiT that the room floor was silent.
   - `MiniMaxH3SoundCues` works around the muting by mixing a deterministic `-28 dBFS` low-passed rumble floor into every island (`floor_dbfs = -28.0`), restoring `-33..-35 dBFS` inter-event ambience. That floor is still a synthetic brown-noise approximation rather than scene-adaptive room tone.

2. **`BasicGuider` still leaks synthetic audio at `audio_end_percent = 0.50` or `0.60`—it needs `0.33`.**
   - In `exp1_sound` Arm 3 (`BasicGuider` at `audio_end_percent = 0.60`, `4/6` steps), `cond_audio` stayed active through step `3` (`sigma_a = 0.50`), baking the synthetic waveform into `sample_lcm`'s running audio trajectory (`r = +0.20..+0.26`).
   - Dropping `audio_end_percent` to **`0.33`** (`Gate 1`, `2/6` steps) eliminated the bleed under `BasicGuider` (`r = -0.1578` stereo / `-0.0078` mono) while keeping all `3/3` hops at `1x` cost (`6` passes). If you want `cond_audio` held through step `2` or `3` (`0.50..0.60`) for extra visual contrast, you must pair it with `MiniMaxH3EditGuider(streams="video_only")` so the audio stream never receives the `cond_audio` pass.

3. **`video_end_percent = 1.0` and `0.50` both freeze leg articulation on 2D sprite cutouts.**
   - Holding `<Video 1>` across all `6` steps (`exp2_dpad` Arm 3, `vep = 1.0`) caused a `+7`-frame jump apex lag (`f56` vs `f49`) and froze the tortoise's legs into `<Picture 1>`'s static standing silhouette across the backward walk (`f68..f92`).
   - Releasing `<Video 1>` at `0.50` (`Gate 2`, `3/6` steps, `sigma_v = 0.80`, `b = 0.25`) fixed the jump lag (`f51..f52`) and kept the `-383 px` backpedal, **but `f72..f92` legs remained mostly frozen** because `75%` of base flow time (`b: 1.00 -> 0.25`) had already elapsed under the static cutout. Only releasing at **`0.33`** (`Gate 2b`, `b = 0.50`) freed leg articulation. There is a real tradeoff here: releasing earlier than `0.33` (e.g., `0.17`, `1` step) risks losing late-clip trajectory reversals (`f66..f112`), whereas `>= 0.50` freezes limbs.

4. **Auto-cutout (`GrabCut`) is a border-seeded heuristic, not semantic segmentation.**
   - When no `mask` is wired, `MiniMaxH3AppendSpriteTrajectory` seeds OpenCV `GrabCut` from a `6%` border strip plus an upper-center bright-feature guard (which keeps tall thin attachments like the white birthday candle from being pruned).
   - That works cleanly when `<Picture 1>` frames a centered subject inside a distinct margin, and fails when the subject fills the frame edge-to-edge or shares color histograms with a busy background. For complex studio or crowd stills, wire an explicit `MASK` (such as SAM 3) into `MiniMaxH3AppendSpriteTrajectory.mask`.

5. **The sprite trajectory is flat 2D affine paste-and-flip—no 3D yaw, foreshortening, or limb cycle.**
   - `MiniMaxH3AppendSpriteTrajectory` translates, scales, and horizontally mirrors a single RGBA cutout over a flat two-tone grey horizon (`0.52` sky / `0.44` ground) with a soft dark ellipse shadow. It cannot represent a smooth `90°` camera-facing turn (`z`-axis rotation), depth occlusion behind scene props, or quadruped gait phase on its own; it relies entirely on H3 hallucinating 3D volume and gait after step `1`.

6. **Sound cues anchor touchdown/impact, not liftoff.**
   - In a preliminary check, placing `hop` at `frame_idx = 48` (mid-flight of `hop 44..54`) pulled the acoustic transient ahead of the visual landing. Percussive cues correspond to physical contact (`54` touchdown for `hop 44..54`), so the user still has to align audio cue frames with the **end** of an aerial arc rather than its start.

7. **3-frame (`125 ms`) Audio VAE grid snapping and `< 200 ms` island merging.**
   - Because `FRAME_RESCALE = 5/3` (`40 Hz` audio tokens / `24 fps` video frames), only multiples of `3` video frames map to exact integer audio tokens (`5/3 * 3k = 5k`). With `snap_to_3f = True`, frame `56` snaps to `57` (`±1` frame = `±41.7 ms`). Furthermore, two cues within `< 5` frames (`~200 ms`) have overlapping waveform tails and are merged into a single `cond_audio` island anchored at the earlier frame so their 1D RoPE coordinates never collide.

8. **Authoring trajectories in multiline text is still manual.**
   - Typing normalized `x=0.31, y=0.69, face=right` lines is deterministic, git-diffable, and needs zero external files, but guessing screen-space `x`/`y` coordinates against a background street described only in text still takes iteration compared to dragging a path over a preview frame.

---

## 6. What else it might also do

The mechanisms verified here are general properties of `minimax_refs`, `minimax_keyframes`, and `MiniMaxH3ReleaseReferences`. The extensions below follow directly from the same code paths (unrendered items are marked as unverified):

1. **Two-subject independent choreography (`<Picture 1>` + `<Picture 2>` -> `<Video 1>` + `<Video 2>`):** *(unrendered, supported by schema)*
   - Because `MiniMaxH3AppendSpriteTrajectory` takes `references` in and emits `references` out, two sprite nodes can be chained after two `MiniMaxH3AppendRefImage` nodes so `<Subject 1>` and `<Subject 2>` follow independent `(x, y, face)` paths (crossing paths, approaching and stopping, or jumping in canon), with `video_end_percent = 0.33` releasing both `<Video 1>` and `<Video 2>` simultaneously while keeping both `<Picture 1>` and `<Picture 2>` at `1.0`. (Watch total packed sequence rows: each `384x224x124` sprite clip adds `+3,108` DiT rows + `504` Qwen rows.)

2. **Rhythmic foley, footsteps, and object hit marks in pure `t2va` (`1x` cost):** *(partially measured via `Gate 1` 3-hop test)*
   - Using `MiniMaxH3SoundCues` (`step`, `click`, `slam`, `snap`, `hop` with `@gain` multipliers and `s` or frame timestamps) + `BasicGuider` + `audio_end_percent = 0.33` adds ~`0s` sampling overhead (`6` DiT passes) and can time door slams, ball bounces, sword parries, or syncopated footsteps without external audio assets.

3. **Early release (`video_end_percent = 0.33..0.50`) on grey-plate motion references in `masked_v2v`:** *(unrendered hypothesis)*
   - The `masked_v2v` lane (`docs/wiki/masked_v2v.md`) already feeds grey-background `<Video 1>` motion references alongside `<Picture 1>` identity stills at `end_percent = 1.0`. Where a low-detail or masked `<Video 1>` over-constrains fine cloth folds or limb secondary motion while `<Picture 1>` carries sharp identity, sweeping `video_end_percent` in `0.33..0.50` while keeping `end_percent = 1.0` tests the same separation on live-action motion plates.

4. **Paired still-keyframe (`i2va` / `fl2va`) + mid-clip sound cues:** *(supported by `PackedLayout`)*
   - In `i2va` or `fl2va`, `first_frame` / `last_frame` sit in `minimax_keyframes` as `cond_video` (`latent` key), while `MiniMaxH3SoundCues` appends `cond_audio` (`audio_latent` key). Setting `end_percent = 1.0` and `audio_end_percent = 0.33` keeps the visual bookend frames locked through step `5` while releasing the mid-clip acoustic beat markers after step `1`.

---

## 7. Under the hood

### 7.1 `PackedLayout` RoPE coordinates and why overlapping `cond_audio` spans fail
Inside `comfy/ldm/minimax/model.py` (`PackedLayout`, lines `293-431`) and `_cond_latents`, H3 packs conditioning latents into non-denoised segments (`img_update = False`, `VISUAL_COND_TIMESTEP = 0.999` on visual cond tokens, `t = 0` on audio cond tokens):

- **Reference media (`minimax_refs`):**
  - `<Picture N>` and `<Video N>` VAE latents (`ref_img`) are placed at **negative** temporal RoPE indices (`t < 0`) immediately preceding frame `0`, separated by spatial/temporal cursor offsets so they carry identity/motion context without claiming a target output timestamp.
  - Because a `124`-frame (`37` latent frames) `1024x576` video reference adds `37 * (576/16) * (1024/16) / 4 = 37 * 576 = +21,312` `ref_img` DiT rows, `MiniMaxH3AppendSpriteTrajectory` renders at **`384x224`** (`224/16 * 384/16 / 4 = 14 * 24 / 4 = 84` tokens/latent frame -> **`37 * 84 = +3,108` rows**, **`-85.4%`**) and sets `RuntimeVideoReference(..., allow_upscale=False)` in [`reference_conditioning.py`](../../../reference_conditioning.py) so `video_policy="release"` never upscales the synthetic plate `11.7x` to `1312x768`.
- **Timeline keyframes (`minimax_keyframes`):**
  - Target video runs at `24 fps` and the Audio VAE runs at **`40 Hz`** (`32,000 Hz / 800` samples = `25 ms` per token), giving `FRAME_RESCALE = 40 / 24 = 5/3` audio tokens per video frame.
  - Each `audio_latent` keyframe of length `rt` anchored at `resolved_frame_index = f` is assigned 1D RoPE coordinates:
    $$t_{\text{aud}}(k) = \left\lfloor \frac{5}{3} f \right\rfloor + k, \quad k \in \{0, \dots, rt - 1\}$$
  - **Why `snap_to_3f = True`:** When $f = 3m$, $\frac{5}{3}(3m) = 5m$ is an exact integer, eliminating up to `16.7 ms` (`2/3` token) floor-truncation jitter between visual frame $f$ and acoustic token $5m$.
  - **Why overlapping islands must merge or be refused:** `PackedLayout` advances no inter-keyframe cursor between `cond_audio` entries—each entry writes directly to $[\lfloor \frac{5}{3} f_i \rfloor, \lfloor \frac{5}{3} f_i \rfloor + rt_i)$. If cue A at `f54` (`rt = 7`, RoPE `[90, 97)`) and cue B at `f57` (`rt = 8`, RoPE `[95, 103)`) were emitted as two separate keyframes, RoPE positions `{95, 96}` would appear twice in `cond_audio` alongside target `audio` `{95, 96}`. [`guide_latent.py::validate_audio_keyframe_spans`](../../../guide_latent.py) and [`preflight.py`](../../../preflight.py) refuse overlapping `cond_audio` spans with a clear `ValueError`, while [`sound_cues.py::merge_sound_cue_islands`](../../../sound_cues.py) automatically mixes adjacent cues into one shared waveform buffer (`[90, 103)`, `13` tokens) before Audio VAE encoding.

### 7.2 Closed-form Pika `ManualSigmas` and why `0.33` is the `b = 0.50` flow midpoint
Pika H3 Flex (`workflows/h3_config.py::PIKA_MANUAL_SIGMAS`) evaluates `6` denoising steps on:

$$\sigma_v = \text{time\_shift}(\,b,\; s_v = 12.0\,) = \frac{12\,b}{1 + 11\,b}, \qquad \sigma_a = \text{time\_shift}(\,b,\; s_a = 3.0\,) = \frac{3\,b}{1 + 2\,b}$$

over exact unshifted linear flow-matching knots **`b = [1.00, 0.75, 0.50, 0.25, 0.10, 0.04, 0.00]`**:

| Step index `i` | Step progress `i / 6` | Unshifted flow time `b` | Video $\sigma_v$ (`shift = 12`) | Audio $\sigma_a$ (`shift = 3`) | Active under `0.33` (`i / 6 < 0.33`) | Active under `0.50` (`i / 6 < 0.50`) | Active under `0.60` (`i / 6 < 0.60`) |
|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| `0` | `0.0000` | **`1.00`** | `1.000000` (`12/12`) | `1.000000` (`3/3`) | **ON** | **ON** | **ON** |
| `1` | `0.1667` | **`0.75`** | `0.972973` (`36/37`) | `0.900000` (`9/10`) | **ON** | **ON** | **ON** |
| **`2`** | **`0.3333`** | **`0.50` (midpoint)** | **`0.923077` (`12/13`)** | **`0.750000` (`3/4`)** | off | **ON** | **ON** |
| `3` | `0.5000` | **`0.25`** | `0.800000` (`12/15`) | `0.500000` (`3/6`) | off | off | **ON** |
| `4` | `0.6667` | **`0.10`** | `0.571429` (`12/21`) | `0.250000` (`3/12`) | off | off | off |
| `5` | `0.8333` | **`0.04`** | `0.333333` (`12/36`) | `0.111111` (`3/27`) | off | off | off |
| *(end)* | `1.0000` | **`0.00`** | `0.000000` | `0.000000` | — | — | — |

In [`control_vector_nodes.py`](../../../control_vector_nodes.py), `MiniMaxH3ReleaseReferences` checks `frac = step_idx / total_steps` on each step:
- **`video_end_percent = 0.33`** keeps `<Video N>` active on steps `{0, 1}` (`b: 1.00 -> 0.50`) and strips `<Video N>`'s `ref_img` tokens (`kind == "video"` / `"video_audio"`) starting at `step_idx = 2` (`frac = 2/6 = 0.3333 >= 0.33`, **`b = 0.50`**), while `end_percent = 1.0` preserves `<Picture N>` (`kind == "image"`) across all `6` steps (`effective_video_end = min(end_percent, video_end_percent)`).
- Because `sample_lcm` (`comfy/k_diffusion/sampling.py:1112`) re-injects fresh Gaussian noise scaled by $\sigma_{i+1}$ after every step, releasing `<Video 1>` at `step_idx = 2` means steps `2, 3, 4, 5` denoise from $\sigma_v = 0.9231 \to 0.0$ with `<Picture 1>` identity intact and zero frozen-leg cutout tokens in `ref_img`, while `orig_layout.position_ids` slices in place with its RoPE `signature` preserved.

### 7.3 `CFGGuider` 3D packed `[B, 1, C_flat]` slicing in `MiniMaxH3EditGuider`
When `SamplerCustomAdvanced` runs on a `NestedTensor(video, audio)`, `CFGGuider.sample` (`comfy/samplers.py:1049-1066`) flattens both tensors into a single 3D packed tensor `[B, 1, C_flat]` before calling `guider.predict_noise(x, timestep, model_options, seed)` and stores `latent_shapes = [(B, 24, T_v, H_v, W_v), (B, 32, 2, T_a)]` on `guider.model_options`.

In [`edit_guidance.py`](../../../edit_guidance.py) (lines `70-135`), `_split_video_audio` checks `isinstance(x, NestedTensor)` first and otherwise calls `comfy.utils.unpack_latents(x, latent_shapes)` so `streams = "video_only"` cleanly applies:

$$\hat{\epsilon}_{\text{video}} = \epsilon_{\text{orig, video}} + s \cdot \bigl(\epsilon_{\text{edit, video}} - \epsilon_{\text{orig, video}}\bigr), \qquad \hat{\epsilon}_{\text{audio}} = \epsilon_{\text{orig, audio}}$$

on steps `0..last_step - 1` (`0..2` for `last_step = 3`, `2` passes/step $\times 3 + 1$ pass/step $\times 3 = 10$ DiT passes total), feeding the clean `orig` prediction to the audio stream so `cond_audio` steers video motion without entering the audio trajectory.

### 7.4 Summary of measured runs (`1344x768x124`, `24 fps`)

#### Audio beat-sheet runs (`unet_fl2va` + Pika `fl2va`, `seed = 102`, target hops `[24, 57, 96]`)

| Arm | Guider & passes | `audio_end_percent` | Cue floor | Detected hops (frames) | Audio `r` vs `.wav` | Inter-event ambience | Wall time |
|---|---|---:|---:|---|---:|---:|---:|
| `exp1` Arm 1 (text only) | `BasicGuider` (`6`) | n/a | n/a | `[23, 55]` (`2/3`, missed `f96`) | `-0.0568` | `-32.98 dBFS` | `~54s` |
| `exp1` Arm 2 (unreleased `.wav`) | `BasicGuider` (`6`) | `1.00` | `-94 dBFS` | `[23, 57, 96]` (`3/3`) | `+0.1636` *(bleed)* | `-52.90 dBFS` *(muted)* | `~54s` |
| `exp1` Arm 3 (`0.60` `BasicGuider`) | `BasicGuider` (`6`) | `0.60` | `-94 dBFS` | `[24, 57, 96]` (`3/3`) | `+0.2036` *(bleed)* | `-32.04 dBFS` | `~54s` |
| `exp1` Arm 5 (shipped `fl2va` `EditGuider`) | `EditGuider video_only 1.6` (`10`) | `0.60` | `-28 dBFS` | `[24, 56/57, 96]` (`3/3`) | **`-0.0086`** | **`-35.58 dBFS`** | `83.9s` |
| **`Gate 1` (`BasicGuider` `0.33`)** | **`BasicGuider` (`6`, `1x`)** | **`0.33`** | **`-28 dBFS`** | **`[24, 56/57, 96]` (`3/3`)** | **`-0.1578` (`-0.0078` mono)** | **`-35.36 dBFS`** | **`60.3s`** |

#### Hybrid `<Picture 1>` + 2D sprite `<Video 1>` + sound cue runs (`unet_ref2va` + Pika `ref2va`, `seed = 104`)

| Arm | Guider & passes | `end` / `video_end` / `audio_end` | Jump apex frame (sprite `f49`) | `cx` trajectory (`f00 -> peak -> f112`) | Backward travel (`f64..f112`) | Facing `+x` & candle lit | `f72..f92` leg motion | Audio `r` vs `.wav` |
|---|---|---|---:|---|---:|---|---|---:|
| `exp2` Arm 1 (`<Picture 1>` + text only) | `BasicGuider` (`6`) | `1.0 / — / —` | `f50` | `451 -> 888 -> 888 px` | **`0 px`** *(fails reversal)* | `+x`, lit | stands still | n/a |
| `exp2` Arm 3 (`dpad_mid_reverse` held) | `BasicGuider` (`6`) | `1.0 / 1.00 / —` | `f56` (`+7f` lag) | `416 -> 843 -> 490 px` | `-353 px` | `+x`, 100% lit | **frozen cutout glide** | n/a |
| `Gate 2` (hybrid `vep = 0.50` + `hop@f54`) | `EditGuider video_only 1.6` (`10`) | `1.0 / 0.50 / 0.60` | `f51..52` (`+2..3f`) | `390 -> 825 -> 442 px` | `-383 px` | `+x`, 100% lit | mostly frozen | `-0.0230` |
| **`Gate 2b` (shipped hybrid `vep = 0.33`)** | **`EditGuider video_only 1.6` (`10`)** | **`1.0 / 0.33 / 0.60`** | **`f51` (`+2f`)** | **`392 -> 816 -> 466 px`** | **`-350 px`** | **`+x`, 100% lit** | **`3` alternating backward stride cycles** | **`-0.0196`** |
