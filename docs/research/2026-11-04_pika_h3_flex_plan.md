# Pika H3 Flex: a plan in priority order, before any decision

2026-11-04. A proposal, not a decision: nothing here runs without the owner's go. The owner's priority among
uses: reference-to-video, then text-to-video, then the masked lane, then first and last frame. The weights
record this rests on is [`2026-11-04_pika_h3_flex_weights_and_loras.md`](2026-11-04_pika_h3_flex_weights_and_loras.md);
section numbers below are that file's. No daily driver is chosen here.

## The question that comes first

The day went into variants (full rank, `svd995`, `svd99`, kijai's resize, spike and tail splits) before anyone
asked whether Pika H3 Flex works in this stack at all, and whether it beats what runs now. Every variant question
sits downstream of that one. Label: **assumption** (that the base case works) and **oversight** (no baseline).
Two renders' worth of Pika's own material can answer it before anything else is built.

## The ledger: what holds, what does not, what nobody knows

**Holds (measured on stored weights or code, section given):**

- The early-access and public adapters are the same bytes (§1). Pika's base components are MiniMax-H3 `42ed227e`
  resharded (§2).
- Pika's Comfy files, Comfy-Org's template files and the fp32 adapter cast to bf16 are the same tensors (§3).
- The fused q|k|v order and swapped `fc1` halves match the int8 convrot checkpoints (§4).
- The delta is a fraction of one int8 step. A merge keeps it only on average, under noise 0.8 to 3.5 times its
  size; `MiniMaxH3LoRABranch` keeps it exactly (§4).
- `lcm` with the manual shift-12 sigmas is Pika's renoise contract for video, and core's audio carry gives
  audio its shift 3 (§6).
- Kijai's resize, done on the fused qkv, loses most of V in blocks 47 to 49 (95% at 49) (§3).
- Each module's change splits into a few strong directions shared by both adapters and a spread-out tail that holds
  what makes fl2va and ref2va differ. Truncation removes tail that is mostly unrelated to that difference (§7).
- On the branch, rank barely changes the cost: full rank is 0.17 s per forward slower than `svd99` at 32768 rows
  (`data/results/2026-11-04_pika_branch_profile_*.json`).

**Bullshit as stated (no evidence, or measured wrong):**

| claim | why it does not stand | label |
|---|---|---|
| `svd995` is the daily driver; `svd99` is a regularizer | No render exists; a smaller file is a hypothesis about quality, not a finding | assumption |
| The top directions are the 6-step distillation backbone | Shared and strong is all the weights show; a shared look or a shared starting point looks the same | not sure yet |
| The switches are a "virtual AdaLN" | True of 11 to 19% of each switch's energy on average; the rest is unexplained (§7) | assumption, partly |
| MLP holds appearance, attention holds motion | Common belief about DiTs, untested here | not sure yet |
| Turn the LoRA off on the last step so base H3 finishes detail | Base H3 is not distilled; its one-jump estimate from sigma 0.333 tends to be soft | assumed wrong, likely |
| Weight-space error measures quality | It measures distance from Pika's training, not the picture | assumption |
| Comfy's templates lose most of Pika | The merge adds noise of known size and unknown effect on the picture | not sure yet |
| Speed should pick the variant | About 0.17 s per forward at 32768 rows; small against everything else in a render | not worth time |
| Compression erases ref2va's task-specific part; audio gets shift 12 | Both measured wrong and retracted (§6, §7) | assumed wrong |

**Unknown, and what each unknown blocks:**

1. Whether this stack reproduces Pika's quality: blocks everything.
2. Whether Pika beats base H3 for each use, and at what cost: blocks adoption.
3. Whether Sol, on by default in every shipped video graph, holds up on a 6-step distill: blocks shipping a graph.
4. Where the look, the few-step finish and the capabilities live, and what each variant amplifies or crushes:
   blocks choosing a variant.
5. How audio fares under any variant: nothing measured.
6. Whether a 6-step renoise sampler works with the masked lane's denoise masks and identity demands: blocks the
   lane.

## Order of operations

Each step names what it decides, roughly what it costs, how likely it is to work and why, and what would stop
the next step. Renders follow the house rules: matched seeds, stacked pairs, the owner's eye on playback, a repeat
and a second shot before anything is called a result, and every prompt read in full from the queued graph.

### 1. Build the graphs (no render)

- Reference-to-video and text-to-video graphs from `workflows/build_workflows.py`. The int8 convrot checkpoint,
  `MiniMaxH3LoRABranch` with Pika's full-rank file at 1.0, `lcm`, `ManualSigmas`
  `1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0`, and `BasicGuider` (no CFG).
- **Reference path:** core's `MiniMaxH3ReferenceToVideo` first, at 1344x768, matching Comfy's Pika template, but
  with two critical fixes verified against `coderef/Pika-H3-Flex`:
  1. **`ref_image_size = "max"` (not Comfy's template default `"match"`)**: Pika's `p768` bucket rule
     (`coderef/Pika-H3-Flex/src/pika_h3/core/buckets.py:52`,
     `coderef/Pika-H3-Flex/src/pika_h3/core/conditioning.py:13-16`) encodes reference images at
     `reference_short_edge = 2048`. In `comfy_extras/nodes_minimax_h3.py:312-319`, `"match"` downscales
     `motion-transfer-input-2.png` (`2816x1584`, a 3-view character sheet) by 4x in area to `1344x768`, whereas
     `"max"` keeps `2816x1584` unshrunk (`1584 <= 2048`).
     **Amended (Claude, same day):** `"max"` is closer but still not Pika's rule. Pika upscales to the 2048 short
     edge by default (`coderef/Pika-H3-Flex/src/pika_h3/core/conditioning.py:142`, "upscale (official
     behaviour)"), while core's `"max"` only scales down (`scale = min(1.0, ...)`). So to match, resize both sheets
     to `3648x2048` before the node, then use `"max"`. (If using the pack's `MiniMaxH3AppendRefImage` +
     `MiniMaxH3ReferenceConditioning`, `size_policy="max", dit_short_edge=2048, allow_upscale=True` is already the
     default in `reference_conditioning.py:876-918`, **but `qwen_view` must be set to `"shared"`** because its
     default `"separate"` shrinks the Qwen3-VL view to `qwen_short_edge=512` at `reference_conditioning.py:948-971`,
     whereas Pika passes the `3648x2048` image to both towers. Conversely, keep `video_policy="comfy"` on
     `MiniMaxH3ReferenceConditioning`: Pika's `conditioning.py:94, 117-120` upscales the `854x480` motion reference
     video to the `p768` canvas `1344x768`, which inflates `ref_vid_rows` from `27,135` to `67,536` and total
     packed rows from `118,867` to `159,268`—1.8x the quadratic dense-attention cost for a `480p` motion-only source.)
  2. **Match each showcase's actual duration/frame count (not a fixed `124` frames)**: `124` frames (`5.17s`) fits
     Comfy-Org's 5-second pilot+station template (`video_pika_h3_flex_ref2va.json`), whereas Pika's showcases are
     longer: `motion-transfer.mp4` is **243 frames (`10.125s`)** (its reference `motion-transfer-input-1.mp4` is
     `241` frames = `10.04s` at `854x480`; running at `length=124` truncates the reference video to `107` frames =
     `4.46s` in `comfy_extras/nodes_minimax_h3.py:337-344`, cutting off the second half of the dance), and
     `balcony-orbit.mp4` is **328 frames (`13.675s`)** with a 5-stage (`0-13s`) camera choreography.
  The pack's typed conditioner (`MiniMaxH3ReferenceConditioning`) comes second, as its own arm. The repo already has
  probe arms comparing the two paths (`workflows/INDEX.md`, `h3_probe_ref_pathway_*`).
- **Attention:** dense to begin with, matching Pika and Comfy, so step 2 tests the pipeline and not Sol. These
  graphs need an entry in `SOL_EXEMPT_STEMS` until step 4 settles it, and the reason has to be written there.
- Decides nothing; makes steps 2 and 3 possible. Likely to work: every part is an existing node.

### 2. Does this stack reproduce Pika? (about 8 renders)

- **Inputs:** Pika's own showcase material (`assets/showcases/` in the early-access download), all at
  1344x768, at their actual frame counts (`243` frames for `motion-transfer`, `328` frames for `balcony-orbit`, or
  `124` frames if testing Comfy-Org's 5s pilot+station template alongside). Two seeds each, with two showcase-specific
  safeguards on `motion-transfer`:
  1. **Settled by the vendor guide (Claude, same day, at the owner's request to follow `vendor_guides/`): leave
     `ref_video_audio_0` unwired. This reverses the amendment just below.** `vendor_guides/ref_en.md` §2.4 makes an
     `<Audio N>` "a standalone audio asset or an enabled synchronized audio track", and §2.5 says "an ordinary
     reference video does not create `<Audio N>` merely because the file contains sound".
     `docs/h3_references.md`, "Labels", says the same of the socket, and adds that a wired reference the prompt never
     uses costs its rows on every step. This target neither copies nor references the source audio (it asks for a
     new soul groove), so the soundtrack has no role, and the vendor format gives it no label. Pika's CLI enables
     soundtracks by default, but that is a pipeline default, not the prompt format. Gemini's original proposal stands.
     The `<Audio 1>` declaration below is then not needed. **Superseded amendment, kept for the record:** wire `ref_video_audio_0` for the reproduction check. Pika's prompt guide says
     a reference video's soundtrack is supplied automatically as `<Audio 1>`, and to "use a video without an audio
     track if you do not want to supply that soundtrack" (`coderef/Pika-H3-Flex/docs/prompt-guidance.md:435, 462`).
     The input has a soundtrack (`mean_volume: -24.2 dB`), so Pika's pipeline (`coderef/Pika-H3-Flex/src/pika_h3/core/conditioning.py:69-72, 379-380`)
     would have fed it in, and the prompt's "Take NOTHING else" is the prompt's job, not the wiring's. Because wiring
     `ref_video_audio_0` emits the `<Audio 1>` label before `<Video 1>` (`comfy_extras/nodes_minimax_h3.py:345-347`),
     the adapted 6-field prompt must declare `<Audio 1>` in `subject_definitions` and mark it as not copied in
     `retention_analysis` (`coderef/Pika-H3-Flex/examples/prompts/ref2va-mixed.txt:1-5`). The original proposal
     follows, kept for the record. **Leave `ref_video_audio_0` unwired**: `motion-transfer-input-1.mp4` carries an embedded audio track (`312`
     frames), while the prompt asks for a new 1960s soul groove and says *"Take NOTHING else from @Video1"*.
     Leaving `ref_video_audio_0` unwired in `MiniMaxH3ReferenceToVideo` cleanly omits `<Audio 1>`.
  2. **Amended (Claude, same day): the adapted 6-field prompt is the main arm. The raw `@Image1` arm runs only if
     the adapted one fails.** "Adapted" means rewritten to the vendor format by the rules in "Prompts, frames and
     references" below, not relabelled. The showcase prompts went through Pika's service rewriting, so the adapted form is
     closer to what the model saw, and at 243 frames each arm is a heavy render. The original proposal follows.
     **Test the 6-field `<Picture 1> / <Picture 2> / <Video 1>` prompt adaptation alongside raw `@Image1` prose**:
     Pika's `coderef/Pika-H3-Flex/docs/showcases.md:22` notes that `assets/showcases/*.txt` are un-rewritten
     service inputs (*"For local generation, adapt the supplied prompts using the prompt guide and preserve
     reference order"*), and both `coderef/Pika-H3-Flex/src/pika_h3/core/conditioning.py:303-310` and
     `comfy_extras/nodes_minimax_h3.py:257-260` prepend literal `<Picture 1>: `, `<Picture 2>: `, `<Video 1>: `
     labels and expect the 6-field `ref2va` schema (`coderef/Pika-H3-Flex/docs/prompt-guidance.md:107-143`).
- **A second arm at no extra design cost:** the same seeds through `LoraLoaderModelOnly`, the template's merge
  path. It settles whether the merge noise shows on screen.
- **Measured cost estimate on this RTX 4090:** with two `3648x2048` sheets (`7,296` rows each), the `226`-frame
  `864x480` reference video (`27,135` rows) and the `243`-frame `1344x768` target (`71,568` video rows),
  `motion-transfer` packs **`118,867` tokens**; `balcony-orbit` at `328` frames packs **`~98,400` tokens**. Against
  the `119,485`-token dense benchmark in [`bench/results/2026-10-03_r2v_finish_time_budget.md`](../../bench/results/2026-10-03_r2v_finish_time_budget.md)
  (`918 ms`/block dense attention = `45.9 s`/step, `~58.5 s`/step total), a 6-step dense render is **~6.1 min** for
  `motion-transfer` and **~4.5 min** for `balcony-orbit` (~42 min for all 8 renders).
- **Decides:** whether to go on. If our renders look clearly worse than Pika's, the fault is in the stack (the
  reference path, preprocessing, prompt form) and every later step waits.
- **Likely:** text-to-video works; this is Comfy-Org's own graph with the branch swapped in. Reference-to-video is
  the real risk, and separating the raw `@Image1` prompt from the adapted 6-field `<Picture i>` prompt isolates
  prompt-format mismatch from reference-pathway issues.

### Prompts, frames and references: the rules every step follows

Added 2026-11-04 at the owner's request: every prompt, frame count and reference in steps 1 to 7 follows
`vendor_guides/` (the only authority on writing an H3 prompt), our reading of it in `docs/prompting.md`,
`docs/h3_references.md`, `docs/h3_resolutions.md`, and, for the masked lane, `docs/masked_v2v/frames_rules.md`. Pika's
own `coderef/Pika-H3-Flex/docs/prompt-guidance.md` uses the same three-field and six-section formats, so the two do not
conflict. Where Pika's CLI behaves differently (soundtracks, reference upscaling), the prompt follows the vendor guide
and the deviation is named.

**Both showcase prompts are raw service text and need a full rewrite.** Neither is in the vendor format: no
sections, `@Image1` instead of labels, and timestamps inside one continuous shot. Pika's `showcases.md` says the same:
adapt them for local use.

**`motion-transfer` (ref2va, six sections, `vendor_guides/ref_en.md`):**
- **Look at the three inputs before writing a word.** Do not assert an attribute you have not seen
  (`docs/prompting/09_references_ref2va.md` §9.11, where "silence is not neutral" is set out). Pika's prose is not a description of the images.
- **`subject_definitions`:**
  - the character as one `<Subject N>` drawn from the character sheet (`<Picture 1>`), with the sheet cited inside
    the definition and no separate picture line (§2.2);
  - the record shop as its own `<Subject N>` from `<Picture 2>`;
  - the dance as its own `<Subject N>` from `<Video 1>`, the form of Pika's Example 3;
  - `<Video 1>`'s own line only for the camera movement and timing it supplies (§2.3).
- **`retention_analysis`:** the character is `attribute_transfer`, with the sheet's views used as identity evidence
  only and the panels never shown (Pika's Example 7). The dance is `attribute_transfer`. `<Video 1>` is
  `attribute_transfer` for the camera only, and says the source person, scene and look are not reused
  (`docs/h3_references.md`, "Transfer motion onto a different subject").
- **`summary`:** opens `[reference generation]`. No audio task type, because no audio is supplied.
- **`detailed_description`:**
  - a style sentence before `[Shot 1]`;
  - 350 to 500 words;
  - the environment described even though a reference supplies it;
  - the duration stated in the prose, 10.125 seconds at 243 frames (`docs/prompting.md` §3.3);
  - the cast count stated positively ("one character, Desmond"), not as a bare negative (§16, "The cast count").
- **Sound:** the turntable's soul groove is heard by the character, so it goes in `detailed_description`, and
  `non_diegetic_music` is `N/A` (`vendor_guides/base_en.md` §4.7). The tapping loafers and record crackle go in
  `overall_soundscape`.

**`balcony-orbit` (T2VA, three fields, `vendor_guides/base_en.md`):**
- **One `[Shot 1]` with no cut.** The five camera phases are written as one path in the closed vocabulary
  (`docs/prompting.md` §4):
  - Pika's "orbit" becomes `Arc Shot`;
  - Pika's "boom" becomes `Pedestal Up` and `Pedestal Down`;
  - then `Push In` and `Pull Out`;
  - phases joined by how one becomes the next, with no in-shot timestamps (§4, "One coherent camera path per shot").
- **Style:** "Live-action, cinematic" with the 35mm look, written at the start of `[Shot 1]`.
- **Cast:** three people, each placed on their balcony.
- **Sound:** `overall_soundscape` carries the wind, city hum, birds and fabric. `non_diegetic_music` is `N/A`.
  No speaker IDs, since no one speaks.
- **Duration:** stated as 13.667 seconds at 328 frames.

**Where prompts live and what checks them:**
- The rewritten prompts are `prompt_bank/` entries, never text in a graph or in code (`AGENTS.md`, "The lanes").
- Each is read in full from the queued graph before it renders (`bench/what_h3_got.py`).
- Each is graded before it renders:
  - `bench/grade_prompt_text.py` and `bench/preflight_graph.py`;
  - `bench/check_ref_prompt_labels.py`, which fails if a wired reference is unnamed or a named one is unwired;
  - `bench/check_prompt_guide_conformance.py`.
- Step 3's prompts for the owner's own shots follow the same rules. Each pair of arms (Pika and base H3) uses one
  identical prompt, so the prompt is not a variable.

**Frames and canvas (`docs/h3_resolutions.md`):**
- 1344x768 is the trained 16:9 canvas and Pika's p768 bucket.
- 243 frames is on the `17n + 5` grid and on both the video and audio clocks.
- 328 frames is on the grid but not the audio clock, which is allowed for standalone text-to-video and matches
  Pika's own output length.
- Both lengths are past the int32 sequence threshold. Dense attention has already run at 119k tokens on this card
  (`bench/results/2026-10-03_r2v_finish_time_budget.md`).

**References (`docs/h3_references.md`):**
- **Video loader:** `force_rate=24` on the reference-video loader ("Frame rate"). This input is already 24 fps, and
  the label check requires it either way.
- **Video truncation:** the reference video is cut to the generated length and snapped down to the grid (241 frames
  become 226).
- **Video size:** core never upscales a reference video. Pika and the other vendor implementations put it on the
  canvas. Keeping core's behaviour (`video_policy=comfy`) is a named deviation, chosen for cost. If step 2 shows the
  dance followed loosely, the release policy is the arm to try.
- **Image size:** sized to Pika's 2048 short edge on both the VAE and text-encoder paths, as section 1 says.

**The masked lane (step 7, `docs/masked_v2v/frames_rules.md` and the lane's own rules):**
- **Windows:**
  - on both clocks: 39, 90, 141, 192, 243, 294 or 345 frames;
  - at least 39 frames;
  - a shorter shot is never split;
  - windows longer than 345 chain.
- **Attention, sampler and steps (corrected the same day):** masked loads are built from
  `h3_config.MASKED_JOB_BASE_GRAPH`, which runs `euler` at 12 steps with Sol live from 20% (`MiniMaxH3Sol`), and no
  masked graph is in `SOL_EXEMPT_STEMS`. This line used to say masked graphs run dense attention, as
  `docs/masked_v2v/frames_rules.md` §0.1 still does. So a Pika masked load is Pika with Sol unless the arm drops Sol,
  and step 4's question does apply.
- **Code needed:** no masked load can take the Pika LoRA, `lcm` or `ManualSigmas` today.
  `workflows/masked_job_graph.py::load` keeps the base graph's model, sampler, steps and shift, and
  `bench/load_chain.py::render_against_plan` refuses a render graph that differs from its plan graph. A Pika masked
  arm needs a load key (with a job flag and a check case) that inserts `MiniMaxH3LoRABranch` and swaps the sampler
  and sigmas.
- **Never unmasked:** the original subject is never shown unmasked in a window.
- **Shared latent steps:** every four frames share one latent step and one mask token.
- **New risks for Pika, untested:**
  - the lane draws faces, mouths and eyes onto `<Video 1>`, and Pika was trained on natural reference videos;
  - the renoise sampler reinjects fresh noise into the whole latent at every step, where the lane's tuning assumed
    base H3's sampler.
- **Not yet read for this plan:** `docs/masked_v2v/reference_and_prompt_rules.md` and the lane `AGENTS.md`. They are
  required reading before step 7.

### 3. Is Pika worth it here? (about 16 renders)

- **Inputs:** the owner's own material, reference-to-video first, then text-to-video. Two shots each, a repeat,
  and matched seeds against base H3's own graphs at their usual step counts: the image-reference graph and the
  text-to-video baseline (`workflows/bench/h3_image_ref_plus_text_to_video_stamped_api.json` and
  `workflows/bench/h3_text_to_video_dense_stamped_api.json`).
- **Record:** wall time, with the cache state it was measured in.
- **Decides:** adopt Pika, per use, or stop. Nothing after this step is worth doing for a use where Pika loses.
- **Likely:** a large speed win (6 steps). Quality is the open question. Distills usually trade some detail and
  prompt-following for speed, and Pika claims extra control (camera, multi-shot, references) that base H3 may not
  match.

### 4. Sol on a 6-step distill (about 4 renders)

- Dense against Sol at matched seeds on step 3's shots, for each use adopted.
- **Decides:** whether Pika graphs keep the house default or stay exempt, with the reason recorded.
- **Likely:** a coin flip. Few steps leave less room to recover from an approximate attention step, but Sol has
  held on other graphs.

### 5a. A first look at the variants (about 8 renders)

Full rank, `svd995` and `svd99` on step 3's reference-to-video shots, with kijai's file as one control. Matched
seeds, a repeat. Added from the Gemini lead's reply the same day, which put this first; this plan keeps it after
steps 2 and 3, because a variant comparison is wasted if the base case fails or loses to base H3. If `svd995` looks
the same, full rank stays the reference: looking the same on a few clips is not the same everywhere, and the
saving is 0.14 s per forward.

### 5b. One activation run, only if 5a or a problem calls for it (no render, one server run)

Moved after 5a at the Gemini lead's suggestion: it earns its cost only if the variants visibly differ or a problem
needs explaining.

- One full-rank run through the branch on real reference-to-video prompts. For each module, step and modality
  (video and audio kept apart), record the small projected Gram matrices (§7's method, checked exact) and how
  constant each switch direction's signal is across tokens.
- **Decides:**
  - whether the switches act as global biases, which would separate calibration from content;
  - where each variant's error lands, including on audio;
  - which attribution arms in step 6 are worth rendering.
- **Needs:** a small tap. It goes in a bench wrapper, not the shared node, unless it is right for every use.
- **Likely:** works; the method is verified. Its value depends on step 3 adopting Pika.

### 6. What lives where: attribution on reference-to-video (about 12 to 20 renders)

- **Arms**, all files the converter already makes, against full rank, chosen by what 5a shows:
  - if full rank over-stylizes or weakens reference identity: a softer body (switches 1.0, body 0.5 to 0.6),
    or a task boost (β 0.25 to 0.5) on ref2va (both the Gemini lead's suggestion);
  - spike only, the most informative single arm about where the look lives;
  - switch strength 0.9 with body 1.0.
- **Dropped:** tail only. Without the top directions it almost certainly will not converge in 6 steps, so it
  teaches little (the Gemini lead's point).
- Two shots, a repeat.
- **Decides:** where the look, the 6-step finish and the reference adherence live, and so which dials are real.
  Then the daily-driver question, with evidence.
- **Most likely to matter:**
  - **Task boost.** It strengthens what makes ref2va differ from fl2va. That is exactly reference behavior, which
    ranks first for the owner. Label: **great idea, not sure yet**. It can overshoot.
  - **Spike only.** Whatever it shows answers the biggest open question.
- **Least likely to matter:** `svd99` against full rank. It removes tail mostly unrelated to the task difference,
  so differences may be too small to see.

### 7. The masked lane (after 2, 3 and 6 on reference-to-video)

- Follows the lane's own rules and protocol (`data/AGENTS.md`, `docs/masked_v2v/PROTOCOL.md`), which this plan
  does not restate and has not checked against.
- **Likely: the lowest confidence of the four uses.** Pika was trained on whole-frame generation, not on
  inpainting a masked region. The lane's constants were tuned on base H3. And identity is the lane's hardest
  demand, which few-step distills tend to weaken. The prize is large if it works: 6 steps instead of the base
  count.

### 8. First and last frame

The owner ranks it last. The fl2va adapter is already exercised by text-to-video in steps 2 and 3, so this
step only adds the endpoint-frame path.

### 9. A daily driver, if one is wanted

Chosen from steps 3 and 6 per use. It may differ by use (full rank for reference work, something else for text
only). Compression only matters if VRAM or host RAM turns out to bind in the masked lane.

## What not to do before step 3

- **More variant files.** Every file built so far came before evidence.
- **Picking a daily driver.**
- **A split-qkv change to `lora_branch.py`.** It is a shared node, and the gain is a fraction of 0.17 s per forward.
- **Step-window or global-strength sweeps.** They depart from Pika's contract and answer nothing step 6 does not
  answer better.

## Alongside

- **Done (`e608d267`):** `bench/convert_pika_lora.py --analyze` now computes the spectrum map, the `fl2va`/`ref2va`
  split, the truncation projection and the AdaLN alignment and writes
  [`data/results/2026-11-04_pika_lora_spectrum_and_adaln.json`](../../data/results/2026-11-04_pika_lora_spectrum_and_adaln.json),
  cited from §7 of [`2026-11-04_pika_h3_flex_weights_and_loras.md`](2026-11-04_pika_h3_flex_weights_and_loras.md).
- **Records.** Each render step gets a record under `bench/results/` (its `.md`), data under `data/`, and video
  under the share's `output/Video/`.
