# Freezing a known audio track: audio-driven video on MiniMax H3

last updated: 2026-09-12

**The owner of this lane.** Opened 2026-09-12 by the owner: drop a song of any
length, keep it exactly, and have the picture move to it, the way the owner's
LTX 2.3 pack (`ComfyUI-AudioLoopHelper`, a sibling custom node, not part of
this repo) does with a per-stream `noise_mask` and an integer-latent loop.
This page says what the mechanism is on H3, what already exists on this box,
the plan in order, and every idea worth a render, each with a confidence.
Nothing here has rendered yet; every claim is a source read, cited, until a
dated record under `bench/results/` says otherwise.

Read first: [`h3_geometry_and_nodes.md`](h3_geometry_and_nodes.md) for the two
clocks and the frame grid, [`h3_references.md`](h3_references.md) "Audio
references" for how core encodes audio and where it crops, and
[`research/sglang_h3_pipeline.md`](research/sglang_h3_pipeline.md) for the
vendor's serving path.

---

## 1. LTX and H3 are different machines, and the freeze means different things

The owner's LTX pack freezes audio by holding one of two streams still while
cross-modal bridges feed it across. H3 has no second stream and no bridge: it
is a single-stream transformer over one packed sequence of text, conditioning
rows, audio rows and video rows, with full attention across all of them
(module docstring, `comfy/ldm/minimax/model.py`). Audio and video share one
time axis through the position ids: `FRAME_RESCALE` audio latent frames per
pixel frame, one position per audio latent frame (`PackedLayout.__init__`,
same file). Each stream has its own sigma shift, video and audio, declared in
the DiT constructor, and the sampler carries the audio latent on the video
schedule through a change of variable (`MiniMaxH3Model.forward`, the
`audio_scale` block).

What a per-stream mask does on H3, all in core (ComfyUI 0.35.0, read
2026-09-12):

- `comfy/samplers.py::CFGGuider.sample` unbinds a nested `noise_mask` into one
  mask per stream and packs them with the latents.
- `comfy/model_base.py::MiniMaxH3._denoise_mask_conds` pools each mask to the
  token grid, quantises to a 1/256 grid, and drops a stream's mask entirely
  when every value is within a thousandth of one.
- `MiniMaxH3.scale_latent_inpaint` re-injects the clean latent into masked
  rows every step: video at the visual conditioning timestep, audio rescaled
  through the schedule carry so the model sees it clean.
- `MiniMaxH3Model._forward` gives each masked row its own timestep,
  `1 - m * sigma`, clamped at the conditioning timestep. A row with mask zero
  is a clean token at its target position, which is how H3 represents its own
  keyframe and reference rows.

So the abstraction is the one the LTX pack uses, and the plumbing is native.
The semantics are H3's own conditioning-row semantics applied to target rows.

**Why it is less of a leap than "zero-shot" suggests.** The audio shift is
smaller than the video shift, so at every step of an ordinary render the audio
rows already sit at a lower sigma than the video rows (`time_shift_sigma`,
`comfy/ldm/minimax/model.py`). A fully clean track is the far end of a
direction the model trains along in every run, not a foreign regime. What the
release did not train is the exact case: sglang's fl2va task profile admits
image keyframes only (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py::MINIMAX_H3_TASK_PROFILES`,
the fl2va row), and DiffSynth's audio-input embedder runs only under
`scheduler.training`
(`coderef/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py::MiniMaxH3Unit_InputAudioEmbedder`).
DiffSynth's inference retake path does what core does, timestep one on masked
rows (`model_fn_minimax_h3`, the `denoise_mask_audio` lines), so both
implementations treat this as inpainting.

---

## 2. Three mechanisms on this box, and a hybrid

| mechanism | where the audio sits | final track | model | trained? |
|---|---|---|---|---|
| **mask freeze** | target audio rows, mask zero | yours, exact if the original waveform is muxed at output | fl2va or ref2va | adjacent: conditioning-row semantics on target rows |
| **reference audio** | reference segment, `<Audio N>: fully_copy` in the prompt ([`prompting.md`](prompting.md) §4.2 vocabulary) | regenerated; sglang derives the target duration from it (`duration_from_audio_reference` in the same task table) | ref2va only; an audio reference cannot stand alone and the budgets cap it ([`h3_references.md`](h3_references.md), the budgets paragraph) | yes, the release's path |
| **guide audio** | keyframe conditioning rows anchored at a frame index (`comfy_extras/nodes_minimax_h3.py::MiniMaxH3AddGuide`, the `audio` input) | regenerated | fl2va | no: a ComfyUI extension with no counterpart in sglang or the release |
| **hybrid** | reference audio *and* the same track frozen in the target rows | yours | ref2va | the reference says what the track is, the mask guarantees it |

**The frozen latent is a control signal, not the deliverable.** The audio VAE
round trip is lossy (DAC-lineage encoder, BigVGAN decoder,
`comfy/ldm/minimax/audio_vae.py`), so the shipped track is the original
waveform muxed at output, as the LTX pack does. Two consequences drive the
ideas in section 5: audio drift in the latent costs nothing in the file, and
the frozen latent need not be the track you ship.

---

## 3. What already exists on this box

Read 2026-09-12. The two packs are installed as siblings of this repo under
ComfyUI's `custom_nodes/`, live in the render server, and are not under
`coderef/` on this date; [`wiki/references.md`](wiki/references.md) does not
yet map them.

- **Core**: the mask path above; `MiniMaxH3AddGuide` for guide audio;
  nested-latent noise and mask packing. Core's `_encode_ref_audio` still
  passes through the generic input crop for the H3 audio VAE
  (`comfy/sd.py` sets `crop_input` false for other VAEs and not this one), so
  a target-audio encode must pad the end to the hop first:
  `reference_conditioning.py::_encode_ref_audio_aligned` does exactly that and
  is the encoder this lane reuses.
- **`ComfyUI-H3-Motion-Context-MultiRef`** (seitanism's fork): a song node
  (`custom_nodes/ComfyUI-H3-Motion-Context-MultiRef/h3_song_audio_context.py::MiniMaxH3SongMaskedAVContext`, a path under ComfyUI, not this repo) that writes a
  master-song slice into the target audio rows, zeros the whole audio mask,
  optionally copies the previous clip's video latent tail in as a frozen
  prefix, and returns the nested mask; an extension node with a short audio
  feather at the seam (`custom_nodes/ComfyUI-H3-Motion-Context-MultiRef/existing_video_extension.py::_apply_audio_context_feather`);
  a mask-editing pair (`MiniMaxH3SetAVNoiseMask`, `MiniMaxH3ClearAVNoiseMask`);
  and the geometry fact that contexts of 39 plus multiples of 51 frames land
  on both the video run grid and the audio grid (`TECHNICAL_ARCHITECTURE.md`
  section 3, derivable from `FRAME_PER_TOKEN` and the audio rate). The
  owner's 2026-08-30 read (`internal/`, gitignored) judged the premise and the
  level of the stack right and the code not worth adopting; its V2V node
  installs a process-wide precision patch that does not retire until restart.
  The song node does not install it. Use the pack as an instrument.
- **`ComfyUI-H3-AudioRefine`**: the inverse, video frozen and audio refined,
  with a KV cache over the frozen rows. The cache does not invert usefully:
  frozen audio is a sliver of the sequence, so a step with frozen audio still
  costs a full forward.
- **This repo**: the aligned encoder above, the audio-carry probe
  (`audio_carry_probe.py`, which understands the schedule carry), the length
  rules (`h3_rules.py`), the generator and `h3_config`. No shipped graph
  writes a `noise_mask` (verified 2026-08-30, and `custom_node_gaps.md`
  section 8 excluded the looping packs from its read).

---

## 4. The plan, in order

Owner agreed 2026-09-12. Each step names what would count as done.

1. **A freeze node on the fl2va base, t2va chain.** Takes the AV latent from
   the existing chain, a song and a start time on the audio grid, slices
   exactly one audio latent's worth of samples per tick, encodes through the
   aligned encoder, writes the target audio rows, and attaches a hard nested
   mask: video ones, audio zeros. The generator emits a variant graph at
   `h3_config.LONG_LENGTH`, which lands on both clocks where the trained
   ceiling (`h3_rules.MAX_LENGTH`) does not (`temporal_shape`,
   `comfy_extras/nodes_minimax_h3.py`). Original waveform muxed at output.
   **Control**: the pack's song node fed the same song and start must produce
   the same latent and mask, bit for bit, apart from its own grid slicing.
2. **Throwaway first run, then one matched-seed pair per scene class**: frozen
   against free audio on a beat-heavy track and on a spoken clip, judged
   free-text before anything is measured ([`eval_comparison.md`](eval_comparison.md)).
   The question: does the picture move to a track it did not generate.
   Section 5's items 1, 2 and 4 fold in here as one-knob variants.
   **Rendering 2026-09-12:** `bench/audio_freeze_step2_arms.json` is the
   arm set (two scenes, `t2va_studio_dancer` against a drum-machine track
   and `t2va_none_of_this_is_real` against a spoken line, each free and
   frozen at one seed, plus the loose-mask arm and the no-transcript arm),
   rows in `bench/results/2026-09-12_audio_freeze_step2_arms.jsonl`. The
   stem variant (item 1) waits on a separated stem; none is on disk.
3. **First-frame keyframe.** The LTX pack's init-image pattern; one graph
   change once step 2 says the mechanism works.
4. **The loop, the LTX geometry on H3's grid.** Window `LONG_LENGTH`, context
   on the joint grid, previous video latent tail copied in as a frozen prefix,
   next song slice frozen, stride equals window minus context, iterations
   through the installed TensorLoop pack. Hard masks on both streams. The
   per-window audio anchor is absolute song time, so nothing accumulates.
   Expect identity drift across windows first; the fix is the ref2va
   checkpoint with a per-window reference image, which is the one reason
   ref2va enters this lane.
5. **Reference audio after the loop runs**, not before: it regenerates the
   track, the opposite of the aim, and its coupling advantage is unproven.
   Then one blind pair as the hybrid. Every idea in section 5 renders
   eventually (owner, 2026-09-12); results decide the order, and this list
   is only the first pass through it.
6. **Feather only when a seam shows.** With the whole track frozen there is
   no audio edge inside a clip; at a loop seam a feather spends ticks of the
   song.

Alongside the code: a decisions line, a `next_steps.md` pointer, and one check
that no shipped graph feeds an AV latent through the stock
`SetLatentNoiseMask`, which replaces the nested mask with a flat one and
silently unfreezes the audio.

**Built 2026-09-12, step 1:** `audio_freeze.py::MiniMaxH3FreezeAudio`, the
graph `workflows/h3_text_to_video_audio_freeze.json` (the generator's
`freeze_audio` knob), and `bench/check_audio_freeze.py`. The control ran the
same day: `bench/audit_audio_freeze_control.py` against the pack's song node
on the real audio VAE, equal to the bit on the audio latent, both masks and
the sliced waveform
([`../bench/results/2026-09-12_audio_freeze_control.json`](../bench/results/2026-09-12_audio_freeze_control.json)).
The throwaway first render's timing row is
[`../bench/results/2026-09-12_audio_freeze_first_run.jsonl`](../bench/results/2026-09-12_audio_freeze_first_run.jsonl);
its clip is a sample, not a result.

---

## 5. Ideas worth a render, with confidence

Each is cheap against what it could change. Confidence is the owner's session
judgement on 2026-09-12, not a measurement.

1. **Freeze a stem, ship the mix.** Freeze a vocal or dialogue stem while
   muxing the full mix, so the model sees speech without drums in the same
   rows; or a click or beat-only stem for a dance piece. Medium confidence,
   very cheap; the largest candidate win for lip and beat coupling.
2. **A slightly loose audio mask.** A mask value around a quarter puts the
   audio rows just below the clean timestep and lets the model own them a
   little, which may couple the video more tightly than rows pinned at one.
   A three-rung ladder answers it. Verify first that the fractional-mask
   x0-conversion defect the 2026-08-30 session found is gone on this build;
   core's forward now applies the mask to the velocity before the carry
   (`MiniMaxH3Model.forward`), which is the fix that PR carried. Low to
   medium confidence, cheap.
3. **The hybrid on ref2va.** Song as a `fully_copy` reference and the same
   song frozen in the target rows. Slots in at the loop stage with the
   checkpoint swap. Medium confidence.
4. **Transcript in the prompt versus not.** With frozen dialogue, lines that
   match the audio may lock mouths to words; lines that do not may produce
   flapping. One pair, and it changes how every dialogue prompt in this lane
   is written ([`prompting.md`](prompting.md) 5.10). High confidence it
   matters, direction unknown.
5. **Two ways to continue a clip.** Frozen latent prefix against the previous
   clip's tail fed as a guide clip through `MiniMaxH3AddGuide`, which is how
   the pack's older continuation worked. Same information, different position
   in the sequence, seams uncompared. Medium confidence either wins.
6. **Guide audio on fl2va.** Core's audio anchor puts the song in
   conditioning rows and lets the target audio generate. Expected to lose to
   the freeze, but it is one graph edit and it says whether H3 copies audio it
   is merely shown, which bears on item 3. Worth trying; the only reason not
   to lead with it is that the target audio regenerates. Low confidence,
   near-zero cost.
7. **Past the trained ceiling, once, with frozen audio.** The ceiling is
   trained, not hard (`h3_rules.py`, the `MAX_LENGTH` note), and a frozen
   track is the strongest anchor the model can get for extrapolating time.
   One step up the grid before the loop exists, expecting failure. Low
   confidence.

8. **Hard freeze on the ref2va checkpoint against fl2va.** Only ref2va
   training ever showed the model clean audio rows at the audio modality
   tag: its reference audio is pinned at timestep one on every step
   (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::MINIMAX_H3_AUDIO_REF_COND_TIMESTEP`),
   while fl2va has no audio conditioning at all. So a frozen target row is
   an index the ref2va weights have seen and the fl2va weights have not.
   If the fl2va freeze looks weak, the same graph on the ref2va checkpoint
   with one image reference is the next arm before any prompt work. This is
   also the argument for idea 2 on fl2va: a loose mask keeps the rows at
   timesteps fl2va saw at its late steps. Medium confidence.

---

## 6. Traps

- **Stock `SetLatentNoiseMask` destroys the nested mask.** It reshapes one
  tensor and stores it flat; the sampler then pads a ones mask for the audio
  stream and the track regenerates. Only a nested mask freezes audio.
- **The trained ceiling does not land on the audio grid; `LONG_LENGTH` does.**
  `temporal_shape` rounds the audio length, so a window that lands on both
  clocks has no rounded tail to reconcile at a seam.
- **Fractional masks are coarse and near-one masks vanish.** The 1/256 grid
  and the drop threshold are in `_token_grid_masks` and
  `_denoise_mask_values`. Hard masks are unaffected.
- **Matched seeds do not match the audio noise across lengths.** Noise for
  both streams comes from one generator consumed in order
  ([`custom_node_gaps.md`](custom_node_gaps.md), the methodological item),
  and with a frozen track the audio draw is consumed and discarded.
- **Two models, one vocabulary.** "Audio" here means DiT-side audio rows.
  Nothing in this lane touches the encoder's handling of audio references.

---

## 7. What would count as finding it

The lane closes as a result, not a build, when a dated record under
`bench/results/` carries: the matched pair from step 2 with the owner's
free-text verdict; the geometry the loop ran at; and, for each idea in
section 5 that rendered, the arm, the seed and the verdict. A rendered clip
that looks right is one sample; two seeds on two scene classes is the floor
([`eval_comparison.md`](eval_comparison.md)).

---

## 8. What the vendor's serving path says

Read 2026-09-12 by a scoped source read of `coderef/sglang` at `b5a2aebc7e`;
[`research/sglang_h3_pipeline.md`](research/sglang_h3_pipeline.md) is the
stage-by-stage authority and this section records only what bears on this
lane. sglang is where the model's authors work, so what it has and what it
lacks are both evidence.

**Reinforced.**

- **Audio rows are always fully visible to every video token.** The Video
  DeltaNet variant windows the video rows and keeps text, condition and audio
  rows dense in both directions (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_vdn.py::VDNH3Layout`,
  `global_ranges`). When the authors chose what attention could be sparse,
  audio to video was not on the list. Coupling runs through attention over
  the packed sequence, which is what a frozen track needs.
- **Frozen means clean and pinned at timestep one, and that is the authors'
  own semantics.** Reference audio rows get no noise at all
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/condition_noise.py`, the `noise_aug == 1.0` early return) and
  keep timestep one every step. The mask freeze puts target rows in the
  same state; the difference is position, below.
- **Audio and video share one origin on the time axis, as core places them.**
  The packed-sequence builder starts target audio and target video at the
  same cursor, one position per audio latent frame and `FRAME_RESCALE` per
  pixel frame (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py`), and a unit test asserts the
  shared origin. A track frozen into the target rows is therefore in
  temporal register with the frames by construction.
- **The two schedules are the ones core derives.** sglang builds video and
  audio sigma grids up front from one base grid with the two shifts, and the
  audio grid equals core's per-step change of variable to floating-point
  precision (the reader checked the nine-point grid). Its scheduler steps
  each stream on its own sigmas with eta zero and draws no noise inside the
  loop (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_minimax_h3_euler_ancestral.py`).
  So freezing one stream disturbs nothing the other stream's update depends
  on.
- **The audio mask hook is half-built in the authors' DiT.** The forward
  accepts an `update_audio_mask` and multiplies the audio logits by it
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py`, the `update_audio_mask`
  lines), but the serving loop never passes one and masks by slicing
  instead, with `skip_mask_out_condition` set. The mechanism is native to
  the authors' code; only the product does not reach it.

**Contradicted, or absent.**

- **No inpainting, retake, mask, extension, continuation, prefix, chunking,
  looping or streaming path exists anywhere in the H3 serving code.** One
  request is one clip through a flat stage list. The pipeline's own reject
  list names trajectory output as unsupported "for its coupled video/audio
  denoise state". The web app exposes audio only as a `reference` condition
  and has no target-track input.
- **Reference audio is not in register with the target.** A reference audio
  block advances the time cursor by its length, and the target audio and
  video start at the new cursor (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py`, the ref2va
  cursor arithmetic). The authors' known-audio path conditions on audio that
  sits *before* the target on the time axis, not on top of it. `fully_copy`
  is a prompt instruction the model follows through attention, not a
  positional alignment. This is the strongest argument in this document for
  the mask freeze over reference audio when sync is the aim, and it is the
  reason the hybrid (idea 3) is worth one pair: the reference states the
  track, the frozen rows give it a position.
- **Keyframes carry no audio, and only first or last frames.** The task
  table admits one keyframe rule per task, image only, and the packed builder
  hardcodes an empty audio conditioning stream. Core's guide-audio segment
  (idea 6) is ComfyUI's construction; nothing in the authors' path suggests
  the checkpoint saw it.
- **The trained ceiling is hard-blocked three times**: the duration constant
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py::MINIMAX_H3_MAX_DURATION_SECONDS`), request
  validation, and the pre-queue probe, with a unit test that a tenth of a
  second past it raises. Idea 7 is outside every validation the authors
  wrote.
- **Noise seeding differs from core.** sglang re-seeds an independent
  generator per stream from the same seed
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/latent_preparation.py`); core consumes one generator in
  order. Neither matters once the audio draw is discarded by the mask, but a
  seed-matched pair across engines is not matched on audio.
