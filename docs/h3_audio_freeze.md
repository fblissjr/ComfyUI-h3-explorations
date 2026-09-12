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

Owner agreed 2026-09-12; rewritten the same evening once the first
verdicts were in. Process note from the owner, same day: this does not need
to be perfect scientific research. Verdicts are the owner's free-text on a
clip or a pair, the instrument sits beside them, and one dated record per
batch is enough.

**Done, and what it found.**

1. **The freeze node, its graph and its control**
   (`audio_freeze.py::MiniMaxH3FreezeAudio`,
   `workflows/h3_text_to_video_audio_freeze.json`,
   `bench/check_audio_freeze.py`; the pack's song node as the bit-for-bit
   control, [`../bench/results/2026-09-12_audio_freeze_control.json`](../bench/results/2026-09-12_audio_freeze_control.json)).
   The node also fixes what it is given (rate, channels, level) and names
   every transform; section 9 checks its path against the vendor's.
2. **Does the picture follow a frozen track: yes, on both scenes**, on the
   owner's word ([`../bench/results/2026-09-12_audio_freeze_step2_verdict.json`](../bench/results/2026-09-12_audio_freeze_step2_verdict.json);
   arms `bench/audio_freeze_step2_arms.json`, rows
   `bench/results/2026-09-12_audio_freeze_step2_arms.jsonl`). The dancer
   "is definitely moving to the beat" at mask 0.0 and "pretty awesome" at
   0.25; the speaker's lip sync "looks good". The mechanism question is
   closed for this lane. What the same batch left open: whether the loose
   mask is better or the seed was (the pair at a second seed,
   `bench/audio_freeze_mask_seed2_arms.json`), and whether the transcript
   needs to be in the prompt (the `voice_untold` arm). The wide framing
   blurs the face, so later renders use `t2va_studio_dancer_close`.

   The lower-face instrument
   ([`../bench/results/2026-09-12_audio_freeze_voice_mouth_own_audio.json`](../bench/results/2026-09-12_audio_freeze_voice_mouth_own_audio.json))
   reads the wrong thing on this scene: the silent tail is where she turns
   her head, so a fixed region sees more motion in silence than in speech
   on every arm, and this OpenCV build has no face detector. The eye is the
   verdict here; the instrument's mouth mode needs a landmark model before
   it says anything.

**Built while the owner was away, 2026-09-12 evening**, for steps 5 and 6:
`audio_freeze.py::MiniMaxH3FreezeAudioWindow` (one window of a long track,
the previous window's video latent tail copied in as frozen context, the
geometry rules in `window_geometry`), the first-frame twin
`workflows/h3_first_frame_to_video_audio_freeze.json`, and the API-only
two-window seam graph `workflows/h3_text_to_video_audio_freeze_2windows.json`
(second window fed the first sampler's output, decoded separately, the
39-frame overlap dropped, the two joined, the muxer on the track's span).
The freeze node also grew a `level` input (the level guard above).

**Next, in order.**

3. **PDD8 as the iteration chain** (owner's ask; the lane's speed lever).
   `bench/audio_freeze_pdd_arms.json` renders both frozen arms on
   `workflows/h3_candidate_t2v_pdd8_baked_audio_freeze.json` at the base
   pairs' seeds. If the PDD renders follow the track as the base ones do,
   every later render in this lane goes through it first and the base chain
   is for the keeper.
4. **The two open pairs from step 2**: loose against frozen at a second
   seed, and transcript against none. Owner's eye, one line each.
5. **First-frame keyframe**: the LTX pack's init-image pattern, one graph
   change on the freeze graph. Needed before the loop, because the loop
   anchors each window on a frame.
6. **The loop**: window `LONG_LENGTH`, context on the joint grid (39 plus
   multiples of 51 frames land on both clocks), the previous window's video
   latent tail copied in as a frozen prefix, the next song slice frozen at
   its absolute time, stride equals window minus context, iterations
   through the installed TensorLoop pack, hard masks on both streams. One
   two-window render of the drum track is the first seam to look at. Expect
   identity drift across windows first; the fix is the ref2va checkpoint
   with a per-window reference image.
7. **Then the ideas in section 5**, ordered by what the verdicts so far
   suggest: the stem (once a separated stem is on disk), the hybrid and the
   ref2va-checkpoint freeze at the loop stage, guide audio and the ceiling
   as single renders when the card is free, feather only when a seam shows.
   Every idea renders eventually (owner); results decide the order.

Alongside: a `next_steps.md` pointer kept current, and the check that no
shipped graph feeds an AV latent through the stock `SetLatentNoiseMask`,
which is in `bench/check_audio_freeze.py`.

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

---

## 9. The audio path, checked against the vendor's

Owner's ask, 2026-09-12: is the track sampled correctly, at the rate,
channel layout and grid the audio VAE expects, as sglang does it. Read the
same day; each row names the observable on both sides.

| step | sglang (reference audio, the only audio it encodes) | this node | agree |
|---|---|---|---|
| decode | ffmpeg to interleaved float PCM at the file's own rate, forced to two channels (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::_load_waveform`) | core's `LoadAudio` decodes with PyAV to float in [-1, 1] at the file's rate (`comfy_extras/nodes_audio.py::load`, `f32_pcm`) | yes: same codec output, different decoder |
| channels | ffmpeg `-ac 2`: mono duplicated, more than two downmixed from the file's layout | mono duplicated; more than two downmixed with ffmpeg's default coefficients under the standard order for that count, the assumed order named in the report (`audio_freeze.py::_stereo`, `_DOWNMIX`); a count with no standard layout averaged. Owner, 2026-09-12: fix it rather than refuse | yes, with the layout assumed here where sglang reads it |
| resample | `torchaudio.transforms.Resample(source, 32000)` at its defaults (`_audio_resampler`) | `torchaudio.functional.resample(wave, source, 32000)` at its defaults (`slice_window`); the transform precomputes the same kernel | yes: same sinc interpolation, same lowpass width and rolloff |
| target rate and grid | 32 kHz, one latent step per 800 samples, 40 steps per second (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_audio_vae/audio_vae.py`, `hop_length` from the encoder rates) | read off the loaded VAE (`audio_grid`: `audio_sample_rate` and `spacial_compression_encode()`), refused if they do not give the 40 Hz grid | yes, and checked at run time rather than assumed |
| length | `-t` caps at the target's duration, then `preprocess` right-pads with zeros to a whole number of hops | exactly `audio_t * hop` samples from a start snapped to the grid; zeros only when the track runs out | yes: the slice is already whole hops, so the VAE's pad is a no-op; core's generic input crop is a no-op for the same reason |
| level | no gain, no peak or loudness normalisation | `level=clip_guard` (default): a DC offset removed, a peak above full scale pulled under it, a well-formed file untouched; `peak` and `none` are the other settings (`audio_freeze.py::condition_level`) | yes on a well-formed file; a clipped or offset file is corrected here and reported, where sglang would encode it as is |
| encoder | encoder, attention projection, `mean_proj`, then `(z - mean) / std` with the stored latent statistics; the posterior mean, no sampling | core's `MiniMaxH3AudioVAE.encode` does the same four steps in the same order (`comfy/ldm/minimax/audio_vae.py`) | yes |
| layout | `[2*T, 32]` rows, channel-major | latent `[1, 32, 2, T]`, packed channel-major by core's `pack_audio` when the sampler runs | yes |
| precision | fp32 | core pins the H3 audio VAE to fp32 (`comfy/sd.py`, the H3 audio VAE block) | yes |

Two things that are not the same and are meant not to be. (The channel
and level rows used to say the node refused more than two channels and
touched nothing; changed the same day at the owner's ask.) sglang's audio
enters as a reference block ahead of the target on the time axis (section
8); ours enters the target rows in register with the frames, which is the
point of the lane. And DiffSynth's retake path pads a short track by
repeating its last latent step; this node pads the waveform with silence
and encodes that, so a track that runs out reads as silence rather than as
a held frame.

The muxed `clip_audio` is the same resampled slice the VAE saw, so the file
carries what the model heard, at the VAE's rate.
