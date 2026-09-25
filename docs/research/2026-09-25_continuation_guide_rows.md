# Continuing a clip by guide rows: six designs and one to build

last updated: 2026-09-25

> **Provenance, 2026-09-25.** A research subagent (Claude) read the sources.
> The session that commissioned it verified the load-bearing points and wrote
> this note: the credits, LongMedia's quote and wrapper, and the licenses.
> Nothing was rendered and nothing was built. Points the subagent took from a
> second-hand read are marked *(second-hand)*.

**The question.** [`../h3_audio_freeze.md`](../h3_audio_freeze.md), section 5
idea 5, asks how to continue a clip. Ours freezes the previous window's
sampled tail as a masked prefix of the target (arm A). The alternative feeds
that tail as clean conditioning rows ahead of a fresh target, regenerates the
overlap and discards it (arm B, guide rows). vllm-omni shipped B on
2026-09-20. This note is evidence for the owner's decision on building B. It
is not a build.

## What changes the picture

1. **Core already expresses B. No core change is needed.**
   `comfy/model_base.py::MiniMaxH3.extra_conds` passes each
   `minimax_keyframes` entry through. `comfy/ldm/minimax/model.py::PackedLayout`
   places an entry `{resolved_frame_index: 0, latent: [1, 24, c, H, W],
   audio_latent: [1, 32, 2, rt]}` on exactly the positions of the target's
   first `c` video steps and first `rt` audio steps. The latent is read with no
   length limit, and its spans restart the 1,4,4,4,4 pattern at the guide's
   first step. `ttulttul`'s pack ships exactly this with no patch
   (`coderef/ComfyUI-Minimax-H3-Continuation/continuation_nodes.py::MiniMaxH3LatentTailGuide`).
2. **A global temporal offset needs no core change either, and it buys
   little.** LongMedia swaps `minimax_payload["layout"]` for a shifted copy in
   a per-clone APPLY_MODEL wrapper
   (`coderef/ComfyUI-MiniMax-H3-LongMedia/temporal_positioning.py::h3_temporal_offset_wrapper`,
   `::apply_h3_temporal_offset`). Core's RoPE enters attention only as a
   position difference, applied to both q and k
   (`comfy/ldm/minimax/model.py::rope_rotation_table`). So a uniform shift of
   the media rows leaves every media-to-media distance unchanged, and those
   are the distances a seam depends on. It moves media only against text and
   stills, further every window. vllm-omni's own recipe warns that global
   positions can exceed the trained temporal range. This is reasoned from the
   code.
3. **The pack vllm-omni credits for the offset argues against B.** In
   `coderef/ComfyUI-MiniMax-H3-LongMedia/nodes.py::MiniMaxH3LatentLabLongMediaNextSegment.prepare`,
   a v0.6.40 comment reads: "continuation authority lives in the actual
   generated target-latent prefix. Do not replace it with a fresh target head
   plus ~22f native keyframe guide; that guide span can become a globally
   repeated motion motif." That is an observation with no record behind it
   that we can read. It is the most specific counter-evidence found.
4. **Under a hard audio freeze, A and B differ in video only, and only a
   little.** Our mask-0 prefix rows and core's guide rows get the same
   timestep, the same conditioning-noise blend, the same modality tag and the
   same positions (`_forward`'s pinned timestep and
   `MiniMaxH3.scale_latent_inpaint` for A, `_cond_video_rows` for B). B adds
   three things:
   - a second, regenerated copy of the overlap at those same positions;
   - under Sol's shipped sink mode, exact-KV treatment of the guide rows,
     where our prefix sits inside the sparse video span
     (`sol_attn_h3.py::_sink_blocks`);
   - more tokens.

## The six designs

| design | tail | where it sits | overlap | positions | audio | licence |
|---|---|---|---|---|---|---|
| **ours** (`audio_freeze.py::MiniMaxH3FreezeAudioWindow`, the song node) | sampled latent | masked prefix, video only | `39+51k` frames, frozen, trimmed at decode | restart per window | locked track at mask 0 | MIT |
| **UtilsCollection loop sampler** (`coderef/ComfyUI-UtilsCollection/helpers/sampling_helpers.py::start_sampling_loop`) | sampled latent, one master latent windowed | masked prefix; audio defaults to partly re-noised | snapped to `5k+2` latent steps | restart | preserve, lock, or regenerate | AGPL-3.0, read only |
| **UtilsCollection re-encode** (`coderef/ComfyUI-UtilsCollection/helpers/model_helpers.py::save_minimax_h3_clip_continuation_media`) | decoded RGB and waveform tails | first and last tail frames as keyframes, tail audio as guide audio, plus a Picture and a text prefix | regenerated, similarity cut | restart | guide rows; target regenerated | AGPL-3.0, read only |
| **vllm-omni #7838** (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/continuation.py::diffuse_continuation`) | sampled cumulative latent | one `latent_guide` block at the target's origin; fresh target | any `17n+5`; regenerated and discarded; one cumulative decode | global offset on media rows | `lock_source` locks the track; `native` regenerates | Apache-2.0 |
| **ttulttul** (`coderef/ComfyUI-Minimax-H3-Continuation/continuation_nodes.py::MiniMaxH3GuidedContinuationWindow`) | sampled latent | guide at index 0, video and audio; fresh target; refuses a competing head guide | `17k+5`; regenerated and discarded; appended; one decode | restart | previous generated audio tail as guide rows; no lock | MIT |
| **LongMedia** (`coderef/ComfyUI-MiniMax-H3-LongMedia/nodes.py::MiniMaxH3LatentLabLongMediaNextSegment.prepare`) | sampled latent | segmented mode: prefix only; multiclip: prefix plus guides *(second-hand)* | frozen prefix, latent splice | global offset via the wrapper; misses `cond_audio` | carried in the prefix, lockable | Apache-2.0 |
| **T8** (`coderef/comfyui-minimax-h3-audio-T8/h3_t8/long_video.py`) | sampled tail | default: guide keyframes at index 0, re-placed by a per-clone `extra_conds` patch; "Plan B" adds a frozen prefix *(second-hand dates)* | regenerated and discarded | restart | reference-audio overlay; lock or remix | GPL-3.0-or-later, read only |

**Provenance.**
- vllm-omni's module docstring credits `ttulttul` for guide/discard/append.
- Its recipe credits LongMedia for the offset convention only. It cites T8
  for a different route: a full-length latent with `remix_source` at 0.
- T8 names both packs as design references
  (`docs/DUAL_MODEL_SEAM_FIX_20260913.md` in its checkout).
- `ttulttul`'s `docs/LEARNINGS.md` records dropping a masked overlap, on the
  premise that H3 "still receives one global timestep". Current core has
  per-row mask timesteps (`comfy/ldm/minimax/model.py::_forward`), so that
  finding does not bear on our hard-mask arm A.

**What can be lifted.** Code may be borrowed, with credit, only from the MIT
and Apache sources: `ttulttul`, LongMedia and vllm-omni. UtilsCollection and
T8 are read only. A node built from this credits `ttulttul` in its
docstring.

## Core's contract, as it stands

- **Multi-frame keyframes are accepted.** The guide uses the target's
  spatial grid, so a canvas mismatch fails at the row scatter. Only
  `comfy_extras/nodes_minimax_h3.py::MiniMaxH3AddGuide` validates anything, so
  a node that writes the dict itself must validate its own inputs.
- **The frame index is not validated.** A negative index would sit among the
  text or reference positions: permitted, untrained, not recommended.
- **There is no per-keyframe strength.** Cond rows run at the global
  `minimax_visual_cond_noise_aug` and `minimax_audio_cond_noise_aug`, which
  move every cond and reference row, stills included.
- **Guide audio** becomes a `cond_audio` segment at timestep 1.0.
  `MiniMaxH3AddGuide` encodes a waveform through the unaligned reference path
  and crops it; a node of ours would write the latent directly.
- **Latent spaces match.** The H3 latent formats carry no scale or shift
  (`comfy/latent_formats.py`), so a sampler output feeds back directly.
- **CFG** would need the guide on the negative too. The shipped chains run
  no CFG.

## A design, if the owner wants B

**`MiniMaxH3FreezeAudioGuideWindow`** is a sibling of
`MiniMaxH3FreezeAudioWindow` with the same inputs and outputs, plus
`positive` in and out, so a graph can swap one for the other.

- `window_geometry` runs unchanged on this window and on `previous`. The
  `39+51k` rule does two jobs here:
  - the phase job: the guide's spans restart at 1,4,4,4,4, so a tail that
    does not start on a phase-0 step would mislabel its steps;
  - the audio-grid job: without it, `slice_window` would round
    `next_start_seconds`.
- The track freezes as today. The video target stays unmasked, and no prefix
  is copied.
- `{resolved_frame_index: 0, latent: previous video tail}` is appended to
  `positive`'s `minimax_keyframes`.
- It refuses:
  - an existing keyframe inside the overlap, since a first-frame image would
    compete with the guide;
  - a canvas mismatch;
  - a missing `previous`.
- `guide_audio` is a combo of `none` (default) and `previous_tail`, per the
  no-zero-sentinel rule. `none` is the default because under a freeze the
  target's overlap audio rows already hold the same track at the same
  positions. A guide copy would duplicate those keys and give the overlap
  audio extra attention weight (reasoned). `previous_tail` is for a
  regenerated-audio mode, which is `ttulttul`'s design and a later pair.
- **A post-sampler splice is needed.** B regenerates the overlap, so a
  per-window decode would take the regenerated overlap, not the real previous
  tail, as the VAE's temporal context. `MiniMaxH3ContinuationSplice` writes
  the previous tail into the sampled latent's head before decode, and
  `trim_frames` then drops it. This is the per-window equivalent of the
  others' single cumulative decode.
- **The song node** would get a `continuation` combo (`masked_prefix`,
  default; `guide_rows`). It stays out of the resume root and folds into the
  keys of windows 2 and later only (`loop_resume.py::window_key`), so a mode
  switch reuses window 1.
- **Overlap: 39 frames**, the smallest context on both clocks and the one
  the owner already judged on arm A. The 22 frames the other packs use is
  off our audio grid.
- **Cost.** B adds the overlap's rows to every window: `video_latent_t` of
  the context against that of the window, at the canvas's rows per step
  (`h3_rules.py`, [`../h3_resolutions.md`](../h3_resolutions.md)). Under Sol
  those rows are exact-KV columns for every query, which erodes Sol's saving.
- **Checkpoint.** A clean multi-step conditioning block ahead of the target
  is outside what fl2va was trained on, since the release's keyframes are
  single images. vllm-omni runs its guide only on ref2va, whose training
  includes multi-step clean reference-video rows. So the checkpoint is a live
  variable. Run the first pair on the lane's fl2va chain; a ref2va pair is
  the follow-up if B looks weak.

*2026-09-25, later: LongMedia's offset is not uniform, so its "global
clock" is partial.* It shifts `cond`, `ref_audio`, `audio` and `video`, and
leaves out `cond_audio` (a keyframe's audio slides against its video) and
`ref_img` (a reference video's frames slide against its own soundtrack). Also,
the offset measurably weakens the prompt's hold on the media
([`2026-09-25_temporal_offset_and_adaln_rounding.md`](2026-09-25_temporal_offset_and_adaln_rounding.md)).
That strengthens "not recommended".

**Needs a core change:** a per-keyframe strength or timestep, a guide on a
different canvas, or a distinct modality tag for guide rows. **Buildable
here but not recommended:** the global offset, and a negative-index guide.

## The pair that would decide it

Route the pair through the `h3-ab-session` skill. It is a practical build
choice, so the owner's free-text verdict on matched pairs is the standard.

1. A throwaway first run of the node and the splice. Confirm in the report
   and in `/history` that the guide rows are present and that window 1 was
   reused.
2. One matched pair per scene class, both arms through the song node's
   combo:
   - the studio dancer on the drum track, which already rendered a
     two-window seam;
   - a dialogue scene whose speech crosses the join, since a join in silence
     tests nothing about sync.
   Then a second seed for each.
3. **Hold fixed:**
   - the PDD8 baked freeze chain, at 1344x768 (or 1152x768);
   - 345-frame windows, context 39, hard freeze at mask 0 with
     `track_latent`;
   - the same prompts, references and per-window seeds;
   - Sol at its shipped settings.

   Only window 2 differs.
4. **The owner judges** three things:
   - seam continuity: a hitch, a pose jump, a lighting flash, or LongMedia's
     repeated motion motif;
   - lip or beat sync across the join, since the file's audio is identical
     under a hard freeze;
   - identity drift, which needs three or more windows and is deferred unless
     the pair ties.

**Arm A already has a positive verdict**
(`../../bench/results/2026-09-12_audio_freeze_step2_verdict.json`), so B has to
beat a working baseline.

## Side finding: LightX2V's causal RefA2V checkpoint

It is published nowhere that could be found. The config holds a placeholder
path (`coderef/LightX2V/configs/minimax_h3_causal/minimax_h3_causal.json`).
The two PRs have no description. Searches of Hugging Face, the lightx2v
organisation and the H3 integration list found nothing. It is not RAVEN
(`mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA`), which differs in form, task and
cache policy. That it was trained with clean driving audio in the target rows
is reasoned from its scheduler and runner, never stated. So
`h3_audio_freeze.md` section 5 item 8 stays plausible and unconfirmed.

## Not established

- A's per-window decode against a splice, bit for bit.
- Whether `bench/preflight_graph.py` prices Sol's sink rows.
- LongMedia's segmented-default claim and its `cond_audio` offset gap
  *(second-hand)*.
- Whether vllm-omni's ref2va-only gate is deliberate.
- When `ttulttul`'s masked-overlap experiments ran, relative to core gaining
  per-row mask timesteps.
