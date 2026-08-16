# Changelog

Semantic versioning. Nothing here has been tagged or published, so every
version below describes the state of the working repo rather than a release
artifact.

## 0.15.0

### Changed

- **Default sampler is now `er_sde`**, replacing `res_multistep`. Owner's
  call; the old value was core's base-template default carried unquestioned.
  One eval per step either way, so it is step-cost-neutral. It is stochastic
  (`s_noise` 1.0), the first such default here, which matters for reading an
  A/B more than for speed. The scheduler stays `simple`: `beta` was tried the
  same day and reverted because it moves two of sixteen steps out of
  Sol-Attn's sparse window (11 sparse / 5 dense under `simple`, 9 / 7 under
  `beta`) with no benefit measured against that. Reasoning lives at `SAMPLING`
  in `workflows/h3_config.py` and nowhere else, same as every other default.
- **`bench/bench_e2e_h3.py` now reads `sampler`/`scheduler` from
  `h3_config.SAMPLING`** instead of hardcoding them. This is the exact shape
  of the bug `check_bench_matches_shipped.py` exists for, one field over --
  that check pins the sage and Sol nodes and says nothing about the sampler,
  so the bench could have gone on sampling a schedule no graph ships. Derived
  rather than checked, so nothing has to notice.

### Added

- **`bench/analyze_morton.py`** -- what Morton reordering does to the
  64-token blocks the Sol-Attn router operates on, with no GPU, no model and
  no clip-watching. Morton cannot affect dense attention at all (attention is
  permutation-equivariant and the permutation is undone after the last block),
  so the only thing it can change is **which tokens share a block**, and that
  is pure arithmetic on the latent grid. Reports per-block frame span,
  bounding box, RMS radius from the block centroid, fill, and the share of a
  token's grid neighbours kept in-block; `--map` prints one latent frame as
  ASCII block ids. The shipped `morton_perm` is imported from
  `vendor/sol_attn_minimax.py` and cross-checked against an independently
  written implementation before any number prints.

  Two findings, both at 294 frames:

  **`morton_curve="2d_frame"` delivers whole 8x8 tiles on only 3 of the 48
  legal landscape canvases** -- 1280x768, 1024x768 and 768x768, the ones whose
  latent dims `(h//32, w//32)` are both multiples of 8. Morton codes tile a
  padded power-of-two space, so a latent grid like 1344x768's 24x42 keeps only
  the in-range corner of each tile. Measured: at 1024x768 a block is a solid
  8x8, fill 1.00, centroid radius 3.24. At **1344x768, the repo's default
  canvas**, a block is typically two disjoint 8x4 fragments in different parts
  of the frame plus a broken right-edge column -- fill 0.60, radius 5.54, and
  4.7% of blocks end up *looser* than raster order's worst block (radius 20.3
  against 16.1). Mean radius still beats raster's 12.12, so morton is not
  simply worse there; it is partial, and its tail is worse.

  **The start-offset rotation in `_perm_for` is load-bearing and correct.**
  Block boundaries are anchored at absolute row 0, so reference rows move
  where the video span's blocks fall; rotating the permutation by
  `(-video_start) % 64` makes block geometry invariant to `video_start`,
  verified at seven offsets. Without it, 1024x768 with references falls from
  fill 1.00 to 0.42. This was nearly recorded as the opposite finding --
  grouping tokens by `j // 64` instead of `(video_start + j) // 64` measures a
  partition that exists on no graph with references, and makes the rotation
  look like the cause of the damage it prevents. `block_ids()` carries the
  correction.

## 0.14.0

### Added

- **`bench/check_sol_kernel.py`** -- the first check here covering a call INTO
  a dependency we do not control. Asserts the installed `comfy_kitchen`
  carries `sol_attn`, that it is the CUDA backend rather than the eager
  reference alone, and that the signature still accepts the kwargs our node
  passes. Presence is gated on a graph wiring `SolAttnMiniMax`, because
  Sol-Attn ships OFF and "absent" is the expected state on any machine that
  has not built the fork; ungated it exits 2, not 0. Shown red against the
  stock PyPI wheel as the control. A second case pins `SOL_CUDA_DEFAULTS`
  against the inputs the node declares, parsed with `ast` so the check needs
  no ComfyUI -- upstream is weighing making `centroid_tail` unconditional,
  and a knob that gets *renamed* rather than removed would otherwise leave
  the pin silently not reaching it while the bench arm kept printing under the
  old name. Shown red by simulating exactly that rename.
- **`custom_nodes/ComfyUI-SolAttn-cuda/`** -- the CUDA node installed
  standalone, upstream's file kept byte-identical with a two-line
  `__init__.py` shim and a README recording provenance. Deliberately NOT
  vendored into this repo: the node id is provisional, so when upstream ships
  the real node this is a directory to delete rather than a graph migration.
  Verified through ComfyUI's own loader path.
- **CUDA arms in `check_solattn_correctness.py`**, grading
  `comfy_kitchen.sol_attn` against the same eager oracle as the Triton
  kernels, with its own red control. Exits 2 when the CUDA arm is skipped for
  cause.
- **`start_percent` sweep arms** in `bench_e2e_h3.py`
  (`shipped+start0.0/0.1/0.3/0.4`), derived from `SOL_RECOMMENDED` so they
  track tau rather than pinning it. It was the only knob in that config with
  no measured rationale -- 0.2 is the paper's number, carried through
  unexamined.
- **`--sol-backend {triton,cuda}`** in `bench_e2e_h3.py`, selecting which
  Sol-Attn node the sol arms build. `triton` stays the default so every
  recorded number stays comparable. The two nodes do not share a vocabulary,
  so `SOL_CUDA_DEFAULTS` is a separate dict in `h3_config.py` and the run
  **refuses** rather than silently dropping an orphaned knob -- otherwise
  `sage+sol+int8` under the CUDA backend becomes plain `sol` and still prints
  as an int8 result.
- **A token-floor warning.** Upstream reports Sol-Attn's gains are invisible
  below ~250-300 frames at 1344x768. `bench_e2e_h3.py` defaults to
  `--length 73` and the frontier table in `docs/SOLATTN.md` was measured at
  124 -- both far under. A Sol arm below the floor now says so, because a null
  result there reads as "this knob does nothing" rather than "this run could
  not have shown anything". Counted in **tokens**, not frames: video tokens
  are `latent_t * (w/32) * (h/32)`, so 250 frames is 72,576 tokens at
  1344x768 but only 44,928 at 832x768. The first version of the guard counted
  frames and would have passed that second run.

### Changed

- **`bench/_sol_attn_reference.py` re-vendored**, `ad9a4a8` -> `c04ef20`. The
  old vendor predated `centroid_tail` (default **True**), which shares one
  pooled tail per query block instead of computing it per row -- so the oracle
  did not contain the path a real render takes, and any correctness verdict
  from it described `centroid_tail=False` only.
- **`check_solattn_correctness.py` now measures which tail mode each kernel is
  on** and grades it against the matching reference, instead of assuming.
  Re-vendoring silently made every Triton arm cross-mode and **they all still
  passed**, because the two modes differ by cos 0.9988 and the bar is 0.998 --
  the bar was looser than a whole-branch change to the algorithm. Measured,
  not read off the kernel: Triton is per-row, the CUDA kernel defaults to
  centroid. Graded in matched modes the two backends' arithmetic is
  equivalent (cuda 0.999919, triton int8 0.999885); the naive cross-mode
  comparison invents a CUDA quality win that is not there.

### Corrected

- **Kernel speed is not end-to-end speed, and this page briefly conflated
  them.** `docs/SOLATTN.md` paired upstream's "CUDA is 1.4x over Triton at the
  same tau" with the `centroid_tail` tooltip's "~1.4x faster" and proposed
  they were the same number, making the backend gap a default gap. The first
  is end to end, the second is the operation. Since e2e speedup can never
  exceed kernel speedup when only the kernel changes, a 1.4x e2e win requires
  a kernel gap well above 1.4x, so a knob worth 1.4x on the op cannot explain
  it -- and upstream puts `centroid_tail` at ~5-10% e2e, which settles it. The
  matching digits were the whole basis of the claim.
- **The 124-frame frontier table measures the wrong regime.** Upstream's floor
  for seeing anything is ~250-300 frames. That is not a weaker version of the
  result; it is a measurement taken where there was nothing to find.

### Notes

- A local build of kijai/comfy-kitchen's `sol_attn` branch is installed as
  `0.2.31+sol.c04ef20`. The branch declares plain `0.2.31`, identical to the
  wheel ComfyUI pins, so nothing could otherwise distinguish the two and a
  `--force-reinstall` would swap the kernel out with no error -- the node
  falls back to dense and the render merely gets slower.
- `SolAttnMiniMax` is **not** wired into any graph, deliberately. Upstream's
  position is that a proper node waits on global attention timestep scheduling
  landing in ComfyUI core, so the node id is provisional, and this repo's one
  rule is that a node id in a saved graph is forever.

## 0.13.0

### Changed

- **Sage runs `fp16 (most accurate)`, not `auto`.** `auto` resolves to the
  *fastest* kernel, which is the wrong end of this project's tradeoff.
  Measured against an fp32 reference across a 17x range of sequence length,
  fp16-PV is **2.7x more accurate and flat** -- no canvas or clip length flips
  the answer -- and the owner judged it clearer with better motion and less
  drift on video at the same seed. Costs ~1.58x per attention call; the
  heaviest shipped config still peaks at 21,186 MiB of 24,564.

  **Both numbers were qualified on 2026-08-14 and the entry above is left as
  written.** The 2.7x is a synthetic-`torch.randn` figure; on q/k/v captured
  from a real H3 forward the fp8-to-fp16 gap is ~1.3x, and the sage fork calls
  every synthetic rtol a pessimistic bound rather than an estimate. The 1.58x
  said "wall clock" here and is a per-call kernel cost -- the fork retracted
  that framing, on the grounds that if attention were nearly all of H3's
  compute a render should run ~1.5x slower and nothing observed shows it. The
  decision stands on its perceptual leg, which is independent of both.
- **Sol-Attn is opt-in and ships OFF.** Bypassed in every UI graph, omitted
  from every API graph. It changes what the model computes, and that has never
  been weighed against what its speed buys. `h3_probe_sol_on.json` is the one
  graph that enables it, so the question stays answerable from an artifact.
  Flipping the default also exposed the split graphs' second model chain
  adding Sol unconditionally, which would have shipped stage 2 enabled while
  everything else was off.
- `check_prompt_guide_conformance.py` now tests the **graph** rather than the
  vocabulary for `keyframe completion`. It rejected the type outright because
  `MiniMaxH3ReferenceToVideo` has no keyframe socket; ComfyUI's new
  `MiniMaxH3AddGuide` supplies one, and `model_base.py` merges its keyframes
  with references additively. The verdict was still right for our graphs, the
  reason was false.

### Fixed

- **The reference-video graphs shipped at a length that does not fit on a
  24 GB card.** Measured, not predicted: at 345 frames the arm builds a
  **182,092-token** sequence (102,816 video, 60,212 references, 16,352 text)
  and the render reached step 4 of 16 at 123.5 s/it before Sol-Attn's kernel
  OOMed and fell back, then sage's OOMed and fell back, then ComfyUI's own
  SDPA OOMed with 21.05 GiB allocated against a 23.54 GiB limit. The fallback
  chain behaved exactly as designed; there was simply no room.

  A reference video is truncated to the **generated** frame count, so its cost
  scales with that number twice over: once for the video rows and once for the
  reference rows. That is why shortening the clip helps disproportionately.

  **The first fix was the wrong lever, and shipping it was a mistake.** Cutting
  the render to 124 frames also cut the reference to 5.2 seconds -- so the arms
  built to measure the expensive case stopped containing it. Re-measured
  best-case-first, the full 345 frames **does** fit once the two incidental
  costs are given up:

  | config | result | peak | time |
  |---|---|---|---|
  | 345f @ 1024x768, refs not upscaled | success | 22,735 MiB / 24,564 | 34.3 min |

  The video-bearing arms now ship at 345 frames on a 4:3 1024x768 canvas with
  reference images left at native size, as `REF_VIDEO_BUDGET` -- one constant
  spread into all eight, so the three numbers have a single home. Canvas and
  reference-image detail are incidental to what these arms measure; reference
  duration is the entire point. 1,829 MiB of headroom is **not** a general
  budget: a third reference image or a longer soundtrack can still exceed it.
- **`MiniMaxH3Preflight` told the user a ceiling was "unreachable at legal
  lengths".** True of length alone, false once references are in play -- which
  is exactly when anyone reads that line. 345 frames plus three reference
  videos reaches 201,246 against a 199,728 wrap, with entirely legal inputs.
  It now reports the remaining headroom and says plainly that length alone
  cannot reach it but references can. Caught by reading Preflight's own output
  on a live render.

- **All fourteen reference arms shipped the same task-type prefix.** Every
  prompt opened `[reference generation]`, hardcoded, including the two edits
  and the continuation -- so `h3_ref_video_edit` and `h3_ref_video_only`
  opened with identical words and the relationship axis those arms exist to
  vary had quietly collapsed. The prefix is now derived from the arm's role
  against the official guide's section 3.2 vocabulary, combined with ` + `
  and never repeated: the edit-plus-images arm now reads
  `[video editing + reference generation + audio reuse]`, which is the
  guide's own worked example verbatim. Motion and structure stay
  `reference generation`, because 3.2 is explicit that presence does not
  imply a type -- only a video actually edited or continued earns its own.
- **The voice arm put its only spoken line in `overall_soundscape`.** Guide
  section 6: dialogue and lyrics go only inside `<d>` in
  `detailed_description`. In the soundscape nothing anchored the line in
  time. It now sits in the shot with its `(S1)` speaker id, and the
  soundscape states only the reference relationship, which is what that
  section is for.
- **The motion arm marked the recipient of a transfer, not its source.**
  `attribute_transfer` is defined as characteristics transferred *to* a
  different subject, so on `<Subject 1>` it read as a request to move that
  person's appearance onto somebody else -- the opposite of the arm's intent.
  `<Subject 1>` is now `fully_preserved` from its image and `<Video 1>`
  carries the transfer. Independently corroborated by two outside
  character-swap examples that mark it the same way.
- **Three Sol-Attn documents quoted a `tau` nobody runs.** The note baked
  into all 26 UI graphs showed `tau=2.0` in its "check it is actually
  running" example, and `docs/SOLATTN.md` showed `tau=1.2 bf16`. The node's
  own default is **1.3** and `SOL_RECOMMENDED` pins 1.3, so anyone following
  the instructions saw a number that disagreed with their own log and had no
  way to tell which was wrong. `SOL_BASELINE_124F`'s 1.2 is untouched -- it
  reproduces old measurements on purpose.
- `docs/SOLATTN.md` located `SOL_RECOMMENDED` in `build_workflows.py` (it is
  in `h3_config.py`) and said it ships a `dense_blocks` starting set (it
  ships `""`; the set is `SOL_ARTIFACT_INSURANCE`, deliberately not wired).

### Added

- **`bench/check_prompt_guide_conformance.py`**, which takes its vocabulary
  from the official guide's own tables at run time rather than from us. This
  exists because `check_ref_prompt_labels.py` rebuilds every prompt the
  generator can produce and compares -- a real guard against hand-edits, and
  structurally unable to catch a generator that is confidently wrong. It
  passed clean through the entire prefix collapse above. The new check
  asserts the six sections and their order, the task-type vocabulary and its
  combining rule, that markers never cross the visual/audio sets, and that
  `<d>` appears only in `detailed_description`. It also rejects
  `keyframe completion`, which is legal vocabulary and inert here:
  `MiniMaxH3ReferenceToVideo` has no keyframe socket and the reference
  implementation drops `image`/`last_image` whenever references are present.
  Exits 2, not 0, when the guide is absent. Shown red six ways, including a
  guide whose tables were reformatted away -- the fail-open case, since every
  other assertion is set membership and membership in an empty set passes.
- **A character-swap arm, `h3_ref_video_swap`** -- the reference video as the
  base plate with its character replaced by one from a reference image. It is
  deliberately the twin of `h3_ref_video_image_edit`: same sockets, same
  budget, opposite request. There the person in `<Video 1>` stays and the
  garment changes; here the person is the only thing that changes.

  Its distinguishing feature is a technique **the official guide does not
  contain**: each reference is told what it does *not* supply (`<Picture 1>`
  supplies identity only, not lighting or background or framing; `<Video 1>`
  does not supply the face). Every relationship in the guide is stated
  positively, so these negative clauses come from general prompting research,
  where the reported failure is the model blending the two identities.
  **Whether they earn their tokens is untested**, which is why the arm exists
  next to its twin rather than replacing it.

  It ships `[video editing + reference generation + audio reuse]`, matching
  the guide's own worked example for this socket combination. Community
  write-ups of this scenario typically stop at a bare `[video editing]`;
  guide section 3.2 adds `audio reuse` whenever the source audio stays
  audible, which here it does at `fully_copy`.
- **`h3_probe_prompt_concise`, the first graph here that breaks the prompt
  format on purpose.** It is `h3_ref_video_swap` with one paragraph instead
  of six sections -- same clip, image, seed, canvas, length and sampler, so
  the only thing reaching the model that differs is the structure. General
  prompting research reports working character swaps from prompts far looser
  than the guides specify, some with no `retention_analysis` and one a single
  sentence. Those reports are uncontrolled, so they are not evidence the
  format is useless, only that nobody has measured it -- and the six sections
  cost tokens in a budget where reference rows already dominate.

  `check_prompt_guide_conformance.py` waives its two structural cases for
  this graph **by name**, prints the waiver on every run, and still enforces
  markers, dialogue placement and label agreement on it. Both were verified
  by mutation: the five conformance cases still go red on a normal graph with
  the waiver in place, and the probe still goes red when its prompt names a
  reference the graph does not wire.

  Fixing this exposed a bug in the day-old check: it located prompts by
  searching for `"subject_definitions:"` in any string input, so the one
  graph with no section headers was skipped **silently** and counted as a
  clean pass over a prompt it never read. It now reads the reference node's
  own `prompt` input.
- `ref_image_count`, so an arm can wire **one** reference image instead of
  always two. The swap arm takes its environment from the plate, so a second
  image was not merely redundant -- it paid reference rows on every sampling
  step to say nothing. `check_ref_prompt_labels.py` caught it as an unnamed
  wired reference before it shipped.
- `VIDEO_ROLES` / `AUDIO_ROLES` are now named once in the generator and
  imported by the drift guard, which previously kept its own hardcoded copy.
  That copy stopped covering the generator the moment `swap` was added and
  reported a freshly generated graph as a hand-edit.
- **The denoising trajectory is now recoverable, not just watchable.** The UI
  graphs gain `GetPreviewOverrideFramesKJ` and a `PreviewImage` sink, which
  return the frames `ModelPreviewOverrideKJ` already decoded through taeh3 as
  an image batch. The live widget shows the current step and forgets the
  previous one; this keeps the whole run, at no extra compute, which is what
  makes two sampler or scheduler arms comparable without paying for a full
  decode each. UI-only, like the preview node itself: in an API graph the
  frames node would not merely be useless, it would raise.
- `force_rate` is now guarded, with the hazard measured rather than argued.
  On three 6.00-second clips trimmed to differ only in frame rate:

  | source | H3 reads it as | error | last conditioner label |
  |---|---|---|---|
  | 24 fps | 5.875s | 0.0% | `<5.2 seconds>` |
  | 25 fps | 5.875s | **+4.2%** | `<5.2 seconds>` |
  | 30 fps | **7.292s** | **+25.0%** | `<7.0 seconds>` |

  At 30 fps the model is told a six-second reference is seven and a quarter
  seconds of action. A 24 fps source is unaffected either way, **which is why
  testing on one proves nothing**. `check_ref_prompt_labels.py` now fails the
  build if any loader feeding a reference socket drops off 24.

### Verified

- **The reference cost model is exact.** Preflight on a live render reported
  `references 60,212`; the model predicted 60,212 from the clip's own
  properties (345 ref frames, latent_t 102, canvas clamped to 960x544 by the
  no-upscale rule, plus two 1024x1024 images upscaled to 2048x2048).
- **The untruncated reference-audio divergence is real, and now observed.**
  Preflight reported `audio refs 1,562` rows = 781 latents = **19.52 seconds**
  of audio conditioning for a **14.375 second** generation. The reference
  pipeline truncates a soundtrack to the generated duration; ComfyUI encodes
  the whole waveform. 412 rows carried past the end of the clip.
- The Preflight duration line, fixed in 0.11.0, is live and correct on a
  reference graph -- the case where it was absent in 7 of 8 graphs before.

## 0.12.0

### Added

- **The reference-combination matrix.** Five graphs differing only in which
  reference sockets are wired: video alone, video plus its soundtrack, images
  plus a standalone audio clip, images plus video plus soundtrack, and all
  four at once. Everything else is shared by construction, so they are
  directly comparable.
- `bench/check_ref_prompt_labels.py`. A `ref2va` prompt refers to its
  references by label, and `MiniMaxH3Tokenizer` derives those labels **from
  the wired sockets, not from the prompt** -- so the two drift silently. The
  check asserts each shipped ref graph's prompt names exactly what its graph
  wires, in the tokenizer's own numbering: images, then videos with each
  soundtrack's `<Audio j>` immediately *before* its `<Video k>`, then
  standalone audio, with a separate 1-based counter per type. A video's
  soundtrack therefore takes `<Audio 1>` and a standalone clip beside it is
  `<Audio 2>`.

  It caught a real mismatch on its first run, and was shown red both ways: a
  prompt naming a label the graph does not wire, and a wired reference the
  prompt never mentions. The second matters because an unreferenced reference
  still costs its rows on every sampling step -- the most expensive way to say
  nothing.
- `_ref_prompt()` generates each arm's prompt from what it wires, following
  `internal/official_prompt_guides/...ref_en.md`: the six sections in order,
  visual markers from section 4.1 and audio markers from 4.2, which are a
  different set and do not interchange.

### Fixed

- **A silent clip cannot have its audio socket wired.** VHS raises "failed to
  extract audio" when its audio output is pulled on a video with no audio
  stream, and the render dies at execution having validated cleanly. The
  video-only arm loads a different, silent clip and leaves the socket alone.
  Found by running it.
- The placeholder input files were checked against `ComfyUI/input`, which on
  this install is **not** the input directory --
  `folder_paths.get_input_directory()` resolves elsewhere. That produced a
  confident "29 of 30 combo entries are stale" conclusion which was entirely
  an artifact of looking in the wrong place. All placeholders verified against
  the real directory; the original two images were correct all along.

## 0.11.0

### Fixed

- **Every API graph in this repo was unsubmittable, and had been since
  `93b08b1` wired the Resolution node in.** `MiniMaxH3Resolution.shape` is a
  `COMFY_DYNAMICCOMBO_V3`, and ComfyUI addresses a DynamicCombo's members by
  their **dotted** path: `shape.wide_resolution`. The generator wrote them
  flat, and ComfyUI's executor answers with `required_input_missing` naming
  `shape.wide_resolution`. The API form is the form `bench/*` drives, so every
  bench run since then was POSTing a prompt the server refuses.

  **`validate_api` had it exactly backwards**, on a belief written into its own
  comment: "the API prompt carries them flat for ComfyUI to re-nest". It does
  not. So the validator was green-lighting graphs the server rejects, and once
  the generator was fixed it briefly rejected the correct form. A validator
  that accepts what the server refuses is worse than no validator.

  Both spellings were tried against a running ComfyUI before either changed:
  dotted accepted, flat refused, for the band case and the `custom` case
  alike. **Found by `bench/smoke_h3.py`, the only thing here that actually
  submits a prompt** -- no static check could have caught it, because the
  static check was the thing that was wrong.
- The three ref `_api` graphs hardcoded `length` while wiring `width`/`height`,
  so sweeping length on `MiniMaxH3Resolution` moved the canvas and left the
  duration behind, silently, and only in the form the benches drive. It also
  skipped the node's own `snap_length()`.
- Four bugs in `reference_fit.py`'s clamp-lift machinery, all one family:
  the "nothing to lift" branch neither armed nor disarmed, so it was the
  branch that let a previous prompt's value through; a **cached** fit node
  never armed at all, so editing only the downstream prompt silently reverted
  to the 2048 clamp with the checkbox still ticked; an unconsumed arm survived
  into a later prompt; and with two fit nodes the one with the box **off**
  cancelled the other's arm, resolved by an execution order that is neither
  the graph's visual order nor settable. Arms are now per node, cleared on
  every downstream call, and `fingerprint_inputs` keeps the node out of the
  cache exactly when the arm depends on it.
- `MiniMaxH3ReferenceFit` now reads the downstream `ref_image_size` from the
  prompt and **says so when it is on `match`** -- under which the stock node
  sizes references from the video's pixel area, never reads the 2048 constant,
  and undoes this node's resize entirely. The log previously reported a 3.6x
  to 16x improvement that had not happened.
- `MiniMaxH3Preflight` printed its duration line only when
  `minimax_frame_count` was present, which core sets **only** on the keyframe
  path. It was therefore absent from 7 of the 8 shipped graphs, including every
  ref graph -- where the 345-frame ceiling matters most. Derived from
  `latent_t` when the key is absent, and marked as derived.
- `preflight.py`'s docstring said the `csrc/fused` uint32 wrap sits near
  199,729; the code computes 199,728.

### Added

- **Reference video, wired for the first time.** `h3_ref_video_to_video.json`
  loads a clip through `VHS_LoadVideo` at **`force_rate=24`** into
  `ref_videos.ref_video_0`, with its own soundtrack into the index-paired
  `ref_video_audios.ref_video_audio_0`. The rate is not optional: the stock
  node has no fps input and assumes 24 twice, for the DiT's temporal clock and
  for the `<T.T seconds>` labels the conditioner reads, so a 30 fps source at
  `force_rate=0` is conditioned 25% slow with nothing said.

  No fit node on this path, deliberately. The same no-upscale divergence
  exists as for images, but a five-second reference at full canvas is +32,256
  rows against +7,168 for a `max` image reference, so it is documented and
  left open until the cost is known to buy something.
- **The two-stage split, both orderings.** `h3_probe_split_base_last.json` and
  `h3_probe_split_base_first.json`: one `BasicScheduler` into `SplitSigmas`,
  two `SamplerCustomAdvanced` stages, `DisableNoise` on the second, and a
  second model chain at an identical shift so the halves run different models
  on one schedule. Built on the custom-sampler stack rather than
  `KSamplerAdvanced`, which was broken on nested latents until core
  `27bca654` (2026-08-12) -- and H3's AV latent is a NestedTensor.
- `bench/check_schema_defaults.py` (from 0.10.1) and six new cases in
  `check_short_edge_override.py` covering the arm fixes above.

### Changed

- `validate_api`'s model-fork invariant understands the split: two model
  sources are allowed **only** when `SplitSigmas` is present, and both stages
  must still read sigmas from one `BasicScheduler`. Verified it still catches
  a stray second source on a non-split graph, and a third source on a split
  one -- a relaxation that could not fail would be worse than the check it
  replaced.
- `MiniMaxH3ReferenceFit`'s second output is `latent_rows`, not
  `vision_tokens`. It returns the DiT's packed rows; the description and the
  module docstring already called it that.

## 0.10.1

### Fixed

- **`MiniMaxH3KeyframeCanvas.execute` still carried the old defaults**
  (`mode="fit_to_canvas"`, `length=0`) after 0.10.0 moved the schema to
  `match_keyframe` / 124. ComfyUI does **not** inject a schema default for an
  input a prompt omits -- the Python signature default is what applies -- so
  the widget fix only protected a node newly dropped in the UI. An
  API-format prompt that left `length` out still emitted 0 from slot 5 and
  rendered the 5-frame, 0.208-second clip 0.10.0 exists to kill, on exactly
  the path `bench/*` drives renders over. Found by review, not by the tests.
- The ultrawide probe's note claimed 1536 is "the widest the trained family
  allows". It is not. Sweeping `adapt_canvas` over the legal 1:4..4:1 range
  gives **eight** canvases at exactly 1008 tokens/frame -- 1344x768,
  1536x672, 1792x576 and 2016x512, plus each transposed -- so the equal-cost
  axis runs to a 3.94:1 frame, not 2.29:1. The note now says so and says this
  probe takes one step along it rather than the last one.
- `_NOTE_TURBO` still read "the trade is aspect: it saw one, where the 4-step
  v0.1 saw mixed aspect ratios" after the table was corrected to show the
  8-step v1.0 as mixed-aspect too. It named the wrong sibling.
- The 0.10.0 "Notes" block described the shipped graphs as not yet
  regenerated. They were regenerated in the same release, so all three of its
  statements had become false.

### Added

- `bench/check_schema_defaults.py`. Asserts every node's `io.Schema` defaults
  match its `execute()` signature defaults, across all 7 nodes, plus that a
  required input never acquires a signature default (which would quietly make
  it optional on the API path). This is the general form of the bug above:
  the two are independent by construction and nothing else compares them.
  Shown red on the real `length` split before being trusted.

### Changed

- `_check_geometry` now documents its own scope. Under the `match_keyframe`
  default an i2v graph derives its canvas from the loaded keyframe at run
  time, so the aspect assertion validates the configured *fallback* rather
  than what renders -- swap in a 3:4 still and the graph renders 768x1344
  having passed a check that looked at 1344x768. The guarantee is not lost,
  it moves to the node, which enforces it on the source image and raises. The
  `GRAPHS` comment's "canvas is shared by construction" claim now carries the
  i2v exception.

## 0.10.0

### Changed

- **`MiniMaxH3KeyframeCanvas.mode` now defaults to `match_keyframe`**, and the
  generator's `canvas_mode` with it. `fit_to_canvas` is the reference
  pipeline's deliberate-override branch, not its default, and it is lossy:
  it cover-crops **both** keyframes, so a 3:4 photo forced to 1344x768 keeps
  43% of its frame. The "it keeps render cost where you put it" defence does
  not hold at the default, because 1344x768 is already the largest area
  `adapt_canvas` ever returns -- it only pays once you lower width/height on
  purpose. `match_keyframe` is also the parity-faithful branch, so anything
  compared against diffusers wants it.
- **`MiniMaxH3KeyframeCanvas.length` now defaults to 124**, from 0. At 0 the
  node forwarded 0, and core's own `min=5` does not catch it: a *linked* input
  skips range validation entirely, so the shipped default rendered a 5-frame,
  0.208-second clip. 124 is the trained floor and matches both core's default
  and `MiniMaxH3Resolution`'s.

  Neither flip touches a saved graph. Widget values are stored per node, so
  only a newly dropped node moves.

### Added

- Both graph builders take `sampler_name` and `scheduler_name` overrides.
  Previously every graph inherited `SAMPLING` with no way to vary the sampler,
  which made the one comparison the vendor's own graphs invite impossible to
  ship.
- `TURBO_768P_LORA` / `_STEPS` / `_SHIFT`, `TURBO_HOME_CANVAS` and
  `TURBO_SAMPLER` in `h3_config.py`, and `check_distill_settings.py` now
  grades **every** turbo triple the config declares rather than the first one,
  with a guard that fails if a new `*_TURBO*_LORA` constant appears ungraded.
  Shown red by setting the 768p shift to 12.
- Six graph variants, from the 08-13 research. None existed before and each
  covers a use case or a theory the shipped set could not express:

  | graph | what it is for |
  |---|---|
  | `h3_text_to_video_turbo_4step_768p` | the only turbo LoRA whose shift is not 12/3, shipped correct rather than described |
  | `h3_probe_turbo_home_canvas` | the 8-step LoRA at the 544p it was distilled at, against the same LoRA at 1344x768 |
  | `h3_probe_turbo_euler` | the vendor's sampler against core's, scheduler held at `simple` |
  | `h3_probe_ref2v_turbo` | ref2v with an fl2v distill LoRA, deliberately out of distribution |
  | `h3_probe_canvas_ultrawide` | 1536x672 |
  | `h3_probe_canvas_portrait` | 768x1344 |

  The last two are the **equal-cost shape control**: 21:9, 16:9 and 9:16 are
  all 1008 tokens/frame, so all three run at an identical sequence length
  (verified: 104,478 at 345 frames for all three) while the long edge goes
  768 to 1536. Every other probe here changes cost to change shape. These
  change shape with cost held constant, which is the only way to ask whether
  the model is shape-neutral rather than just cheaper in one orientation.

### Notes

- The shipped `workflows/*.json` were regenerated against a live ComfyUI in
  the same release. `/object_info` was checked first and reported the new
  `match_keyframe` / `124` defaults, confirming the pack had reloaded --
  regenerating against a stale server bakes in exactly the mismatch
  `check_workflow_schema.py` exists to catch. 31 graphs written and
  validated, UI/API cross-check passing.

## 0.9.0

### Added

- `bench/check_distill_settings.py`. The three FL2VA turbo LoRAs do not share
  a schedule -- two were distilled at video shift 12, the 768p 4-step at 6 --
  and a LoRA inherits the sampler's shift rather than carrying its own. So
  loading the 768p one into a graph still reading 12/3 samples it off a
  schedule it never saw, and nothing errors.

  Covers **every** shipped graph, not only the two that load a LoRA: a turbo
  graph must match its LoRA's row, a base graph must sit at the base
  checkpoint's own 12/3, and the UI and API forms of each graph are paired and
  compared, since they are generated separately and have already diverged once.
  Both the shifts and the recommended step counts are graded against the vendor
  README in `coderef/` rather than against themselves; grading only the shifts
  would leave the step sets self-checked. When `coderef/` is absent that control
  is skipped and the script **exits 2, not 0**, so a runner keying on the exit
  code can tell a skipped control from a clean pass.

  LoRA filenames are parsed structurally rather than by substring, because a
  substring match reads a hypothetical `turbo_8step_v1.0_768p` as the 12/3
  `turbo_8step_v1.0` row -- the exact silent failure the file exists to catch,
  committed by the file itself.

  Shown red eight ways before being trusted: config shift wrong, config steps
  wrong, a turbo graph's shift edited, a base graph's shift edited, the UI and
  API forms disagreeing, our shifts disagreeing with the vendor, our steps
  disagreeing with the vendor, and `classify` reverted to substring matching.
  Green restored after each.
- `docs/checks.md`, an index of every check: what it defends, what it needs to
  run, and whether it has been shown red. Ten of twelve have no such record,
  which is a finding rather than a formatting gap.
- `docs/h3_ref2v_distillation.md`. lightx2v has shipped three FL2VA turbo
  LoRAs and no ref2v one, and their roadmap lists it as future work. This is
  why, from the code. Three mechanisms: fl2v conditioning is positionally
  *identical* to the target (a first-frame keyframe's rotary coordinates are
  `torch.equal` to the target's first latent frame) while a reference sits on
  its own grid and pushes the target's origin by 1 to 1206 units; ref2v is a
  separate `transformer_ref` partition measuring **4.2% relative Frobenius**
  from fl2va while the whole 8-step turbo LoRA measures **0.036%**; and the
  DMD trainer that produced those LoRAs has no ref2v path at all, rejecting
  non-text conditioning outright.

  Both headline measurements were reproduced independently before shipping:
  key sets identical (0 on either side, 1082 shared), projection deltas
  0.037-0.046, `final_layer.adaln_proj` at 1.92 i.e. essentially rewritten,
  and mean LoRA perturbation 0.00036 across 208 touched modules at
  `alpha/rank = 0.0625`. The LoRA does not touch `final_layer`, `adaln_proj`,
  the norms or the patch projections, which is where the checkpoints differ
  most -- a point in favour of the out-of-distribution experiment working at
  all.

  Also records three hypotheses that did **not** survive: re-injected
  reference rows are byte-identical in both tasks, no guidance mechanism
  exists for either, and DMD here is genuinely data-free so "reference pairs
  are scarce" does not bite.

### Changed

- `docs/h3_geometry_and_nodes.md` corrected in three places. The low-VRAM
  saving is **~3227 MiB at 4 groups**, not the ~1070 carried here and in the
  shipped graph notes -- `workflows/h3_config.py` measured three times the
  earlier estimate. `MiniMaxH3SigmaShift` was listed as an untested
  third-party node; it is core ComfyUI and sits in all eight shipped graphs.
  And the keyframe section no longer tells the reader to wire both image
  outputs: with no `last_frame` input, that slot returns the same tensor as
  `first_frame`, so wiring it silently anchors the render to return to its
  opening frame.
- `workflows/build_workflows.py` carried the same stale ~1070 MiB in its note
  template, so it was baked into all eight shipped graphs. Fixed at the
  generator. **The shipped `workflows/*.json` still carry the old number until
  they are regenerated against a live ComfyUI**, which this change does not do.
- `README.md`'s `bench/` listing was missing four scripts and understated what
  needs CUDA or `PYTHONPATH`. It now points at `docs/checks.md` as the index.
- `docs/open_experiments.md` notes which entries the 2026-08-13 plan
  schedules, and which stay blocked on owner judgment.

### Notes

- Every check was run against ComfyUI `12666983` (v0.32.0), comfy-kitchen
  0.2.31 and KJNodes `6ab7e81` on 2026-08-13. All eleven runnable ones pass;
  `check_workflow_schema.py` needs a live server and was not run. Nothing was
  stale in the sense of failing -- the staleness was in the documentation
  around them.
- **Decided: do not reimplement KJNodes' low-VRAM path.** Their node does
  three things and we already own head chunking. Their attention patch yields
  to ours; their block patch is unconditional and unguarded, so writing our
  own block-level release would collide on
  `diffusion_model.blocks.{idx}.forward` with no marker convention. The
  interop cases in `check_lowvram_handoff.py` stay, and that file's name
  undersells it -- two of its five cases guard our own head-chunk
  reassembly, not the KJNodes boundary.

## 0.8.2

### Changed

- `PublisherId` removed from `pyproject.toml`. It exists to publish this to
  ComfyUI's registry, and this repo is not asking to be installed -- the same
  reason the graphs carry no `cnr_id`. Public so the work is readable, not so
  it is distributable.

## 0.8.1

### Changed

- Graphs no longer carry `cnr_id`, reversing a change made earlier the same
  day. It exists so ComfyUI-Manager can offer "install missing custom nodes"
  to someone opening a graph without this pack -- an audience pulling from a
  public registry, which is not what this repo is. Meanwhile
  `useConflictDetection` ships in the same lazily-loaded chunk as
  `useComfyRegistryService` (baseURL `https://api.comfy.org`) and the
  consuming path was not proven to stay local. Under a local-only
  constraint, unproven beats unlikely when the benefit is near zero.
- The workflow `id` namespace seed is a bare string rather than a fabricated
  `github.com` URL naming a handle and a repository. Determinism was the only
  property needed. Every graph id changes once and is stable after.
- Recorded in the generator, so neither is re-added: never emit `models[]`
  (it carries download URLs and the local model directory layout), never
  emit `extra.info` (identity), and never derive `aux_id` automatically --
  its conventional value is the git remote's `owner/repo`, and this repo's
  remote is a LAN address, so that would write a private IP into every
  shared workflow.

### Verified

- Schema: staying on `version: 0.4`. The installed frontend (1.48.7) writes
  0.4 exclusively across every saved workflow on this machine, there is no
  schema-1 serializer in the bundle, and ComfyUI's Python side never parses
  workflow schema at all -- UI workflows move through `userdata` as opaque
  blobs. Moving to schema 1 gains nothing observable and produces a file the
  installed frontend has no writer for.
- Pre-publish scan of tracked content: no home paths, usernames, hostnames,
  LAN addresses, emails or keys. The only IPv4 matches are `127.0.0.1`
  defaults. Our graphs leak less than a frontend save, which carries
  `extra.frontendVersion` and a per-node `ver`; we emit neither and should
  not start.

## 0.8.0

### Added

- `MiniMaxH3Resolution` is wired into every graph except first-frame, where
  the keyframe decides the geometry and `MiniMaxH3KeyframeCanvas` is the node
  that does it. Width, height and length now arrive as links from a node whose
  dropdown says what the choice costs, instead of as two numbers that say
  nothing: `1344x768  7/4  1008 tok/frame  1.00x`.
- Both validators learned that a `DynamicCombo` declares its option inputs
  nested under `options` rather than as top-level inputs. `validate_api`
  called them unknown; `check_workflow_schema` counted the widgets short.
  Third and fourth instance of the same shape -- a node whose input set is
  not fully static -- after VHS's format widgets and dynamic member slots.
  The UI branch was shown red by removing it and watching the count mismatch
  return.

## 0.7.3

### Fixed

- Output prefixes were split across `Video/` and `video/`, which is two
  sibling directories on a case-sensitive filesystem for what reads as one
  destination. All eight graphs now write to `Video/`.
- Our nodes carry `cnr_id` in `properties`, which is what ComfyUI-Manager
  reads to offer "install missing custom nodes". Without it someone opening
  a shipped graph without this pack got red boxes and no way to resolve
  them. Stamped only on nodes this pack owns; claiming another pack's id
  would send Manager after the wrong repo.
- Graphs carry `extra.ds`, so they open on the nodes. Without it litegraph
  starts at its default viewport while these graphs begin at x = -2860, so
  the first thing you saw was empty canvas.

## 0.7.2

### Changed

- Every per-call VRAM figure in this repo is now scoped as per-call, in
  `attention.py` and in `bench_minimax_attn.py`'s own docstring, because an
  e2e run contradicted the inference everyone was drawing from them. At 124
  frames, head_chunks=4 measured 1186 MiB *higher* on process peak than
  head_chunks=1, the opposite direction from the per-call numbers, and a
  second run measured a 2265 MiB spread across two runs of one unchanged
  configuration. The excursion is larger than the effect, so the sign is not
  settled either way at that sample size -- what is settled is that per-call
  peak does not predict process peak.
- `bench_minimax_attn.py` reports `reserved` beside `allocated`. That was to
  test whether chunking fragments the allocator, since reserved rather than
  allocated is what a process occupies. It does not: the two track within
  8 MiB on all three arms, so a single call on a clean allocator does not
  fragment and the mechanism lives in the interaction across 50 blocks and
  every step with a model resident -- which a per-module bench excludes by
  construction.
- The clone still ships. It is free (0.7% wall-clock, bit-identical output)
  and it does lower the attention call's peak. It is simply not demonstrated
  to lower what a user's card reports, and nothing should be spent to get it.

### Fixed

- Workflow `id` is a UUID rather than a readable slug, matching what the
  frontend writes. Stale copies of three graphs in ComfyUI's own workflow
  directory, carrying the old slug ids, were the source of "Unable to find
  workflow in <file>"; they were moved aside rather than deleted.

## 0.7.1

### Fixed

- `MiniMaxH3Resolution` ignored every widget and always returned 1344x768. A
  `DynamicCombo` arrives as one nested dict -- the selected key under the
  input's own id, the chosen option's inputs alongside it -- not as flattened
  kwargs. It read `shape.get("key")` and took its values from `**kw`, so every
  selection fell through to the custom branch and picked up the literal
  defaults. Its test passed because the test called `execute` with flattened
  kwargs: the caller was invented rather than observed, so the test agreed
  with the bug.
- The armed short-edge override could outlive its prompt. Arming is per fit
  node and consumption is per downstream call, and a graph has two fit nodes
  feeding one `ReferenceToVideo`, so the second arm survived into whatever ran
  next. Each fit node now disarms on entry when the flag is off, and a
  mismatched second arm warns.
- `_install_wrapper` would install `classmethod(_make_wrapper(None))` if
  upstream ever moved `execute` to a base class, killing every reference
  render in the process including graphs that never enabled the flag. It now
  declines and says why.
- `SageChainAssert` was emitted with `warn_only=False` even for `sage=False`,
  so a control arm would always raise at the gate. `warn_only` now follows
  `sage`.
- The display renames reached the nodes but not the shipped in-graph note,
  the README, or `docs/h3_geometry_and_nodes.md`, which still told readers to
  search for names the menu no longer has. Two notes in the same graph
  disagreed.

### Changed

- `SageChainAssert` finds Sol-Attn's counters by capability rather than by
  module name. ComfyUI registers a directory custom node under a path-derived
  key, so `import_module("ComfyUI-SolAttn_triton")` resolved in a shell and
  never inside ComfyUI: the call-time probe had never executed once since the
  node was written, and every render printed `chain assert ok` with its
  strongest evidence skipped.
- The override declines an unexpected `q.ndim` instead of raising. It exists
  to catch calls another patch handed back, so the case where a safety net
  earns its keep is a caller nobody predicted -- and it killed the render
  instead of degrading. Our own probe was that caller.

## 0.7.0

### Added

- Three probe graphs, each a pair with a named twin, one variable changed, and
  a note saying what to compare and what to expect.
  `h3_probe_reference_upscale` runs references without the released
  pipeline's upscale, against roughly 13,900 fewer vision tokens.
  `h3_probe_square_canvas` renders the same prompt at 768x768, which is about
  a third of the attention cost and the largest lever anywhere in this
  pipeline. `h3_probe_head_chunks` runs the heads in 4 groups, trading kernel
  launches for 217 MiB of peak.
- `bench/check_generator_constants.py`. Sharing a constant prevents drift;
  this prevents un-sharing, which is a different failure on a different
  timeline -- someone inlining the literal back because it reads more
  directly. Agreement is not a testable property, so each case forces a
  disagreement by moving upstream's value and asserting the graph follows.
  Writing `2048` back into the builder turns it red.

### Changed

- `SEED` is 730451892 rather than 1. Fixed rather than randomised, because
  the probe graphs are pairs and a seed that moved between them would put the
  difference you are looking for underneath the difference you are not.

## 0.6.1

### Added

- `MiniMaxH3ReferenceFit` is wired into both reference graphs, one node per
  reference image, paired with `ref_image_size` on `max`. The pairing is
  load-bearing rather than tidy: under `match` the stock node sizes
  references from the video's pixel area and never reads the 2048 constant,
  so the fit nodes would be silently undone and their two resamples wasted.
  Closes the gap where the graphs named for references were the ones
  under-sizing them.
- `MiniMaxH3Preflight` sits between conditioning and the sampler in every
  render graph, with both consumers reading through it, so a graph cannot
  half-apply it.

## 0.6.0

### Added

- `MiniMaxH3Preflight`. Reads the assembled conditioning and latent through
  the model's own `PackedLayout`, so the sequence length it reports is the one
  attention will run at. Draws on the node: resolution and whether it is
  inside the trained family, sequence length broken down by segment with
  bars, what other aspect ratios would cost at the same length and
  conditioning, and where the int32 thresholds sit for the fused layout.
  Pass-through. It is the only node that can answer "what am I actually
  sending", because the total does not exist until conditioning is assembled.

### Changed

- `MiniMaxH3ReferenceFit`'s `mode` combo is now an `allow_upscale` boolean.
  The two modes differed by exactly `min(1.0, full)` versus `full`, so they
  were one strategy with the clamp on or off, and `reference`/`down_only`
  named provenance rather than behaviour. Saved graphs carrying the old
  string value will need it reset; the node is in no shipped workflow.
- Display names: "MiniMax H3 Keyframe Resolution" and "MiniMax H3 Reference
  Resolution", so the pair reads as two answers to one question. Both
  `node_id`s are unchanged.
- `lift_downstream_clamp` now announces itself as experimental in its label,
  leads its tooltip with what it will cost you, and logs a WARNING rather
  than an info line when armed. It monkeypatches a core node and pushes
  references off the distribution the checkpoint was trained on; that should
  be impossible to trip over by accident.

## 0.5.1

### Added

- `lift_downstream_clamp` on `MiniMaxH3ReferenceFit`, appended last so saved
  widget order is untouched. `MiniMaxH3ReferenceToVideo` sizes references
  with `min(1.0, 2048/short_edge)`, so anything larger this node produces is
  scaled straight back and `short_edge` above 2048 appears to do nothing.
  This lifts that clamp for exactly one downstream call by rebinding the
  module constant the stock node reads at call time, then restoring it in a
  `finally`. Off by default, and above 2048 is off-distribution: 2048 is what
  the released checkpoint conditioned image references at.
- `bench/check_short_edge_override.py` pins the scope rather than the
  arithmetic, because a global rebind that outlived its call would change
  references in graphs that never asked for it. Four properties: applies to
  exactly one call, restores on a raise, declines under `ref_image_size`
  `match` where the constant is never read, and installs idempotently behind
  a marker the way the chaining packs do. Driven through `_make_wrapper`
  with a stub, so it needs no VAE, CLIP or model. Mutated three ways --
  never clearing the arm, dropping the `finally`, removing the marker check
  -- and each turns cases red.

## 0.5.0

### Added

- `MiniMaxH3Resolution`. Pick a resolution by shape and read its cost in the
  dropdown you pick it from: each entry carries resolution, aspect ratio,
  video tokens per latent frame, and attention cost against 16:9. A
  `DynamicCombo` bands the choices so no list runs long -- 22 entries at
  worst -- while all 95 trained resolutions stay reachable, plus `custom`
  for anything the DiT can patch. The list is swept from `adapt_canvas` at
  import rather than hardcoded, so it tracks upstream.
- It answers the question no other node could: whether you are inside the
  trained family. Core's conditioning nodes never call `adapt_canvas`, so the
  768 short edge and the area cap constrain nothing you type. 1024x1024 is
  legal, renders, and is outside the family; the node says so and names the
  resolution `adapt_canvas` would have given instead.
- A README section on which sizing node to use when, built on the rule that
  separates them: a keyframe is patchified on the video's latent grid so its
  resolution must equal the video's, a reference is patchified on its own so
  its resolution only sets the vision tokens it contributes.

## 0.4.3

### Added

- `SageChainAssert` is now last in the model chain of every render graph,
  after Sol-Attn, asserting what the composition ended up as rather than what
  any one node intended. That seam is negotiated through a duck-typed
  attribute both sides rewrote within a minute of each other once already,
  and when it breaks the render still succeeds while being quietly slower or
  numerically different. `exercise` stays on, so the evidence is call-time
  rather than install-time: a fraction of a second and ~176 MiB transiently,
  against renders that peak near 3 GB in attention alone.

## 0.4.2

### Fixed

- `MiniMaxH3ReferenceFit` over-reported its cost by 4x. It counted VAE latent
  cells, where the DiT patchifies those 2x2 before attending them, so a
  reference fitted to 2048x2048 reports 4096 vision tokens and not 16384.
  The 0.3.0 entry's "1024 latent rows instead of 16384" was the same error:
  the real figures are 256 and 4096.
- `check_reference_fit.py` imported the node's own `_rows` and graded it
  against itself, so it passed while every number was 4x high. It now checks
  against `comfy.ldm.minimax.model._frame_grid`, which is what actually
  builds the reference block. Reverting the math turns 7 cases red.

### Changed

- That output is now `vision_tokens` rather than `latent_rows`, matching
  standard transformer terminology: tokens qualified by modality, sequence
  length for the total.

## 0.4.1

### Added

- `check_no_write_through_to_source`: patching a model must not reach the
  model the caller still holds. It runs against a deliberately
  shallow-cloning fake, because against the real `ModelPatcher` the
  assertion cannot fail and would be decoration. Under the real thing the
  node's defensive copy is redundant; the case pins that the node does not
  depend on that, since the failure it prevents is a contaminated A/B
  control arm that looks like a result.
- The `ModelPatcher` stub now clones the way ComfyUI does, ported from
  ComfyUI-AudioLoopHelper's `tests/_fakes.py`. It previously returned
  `self`, which cannot tell a node that mutates its source from one that
  does not.

### Fixed

- The comment explaining that copy said `clone()` shallow-copies
  `model_options`. It does not, and did not when the comment was written:
  `clone()` runs them through `comfy.utils.deepcopy_list_dict` on every
  branch, which landed 2026-02-10. No behaviour was wrong, only the stated
  reason, which is the kind of thing a later decision gets made on.

## 0.4.0

### Changed

- All graphs write video through `VHS_VideoCombine` instead of
  `CreateVideo` -> `SaveVideo`. One node, muxing the audio itself, at 24 fps,
  `video/h264-mp4` (yuv420p, crf 19), `save_metadata` on, `save_output` on,
  prefix `Video/h3_<task>`. h264 rather than h265 or nvenc: software x264 at
  crf 19 is the most portable mp4, and the nvenc paths trade quality per bit
  for encode speed on a file that writes in seconds next to a render that
  takes minutes. `trim_to_audio` stays off, since H3 generates the pair
  jointly and trimming video to the audio track can only drop frames it meant
  to keep.
- The canvas note is rebuilt around the rule rather than examples: you pick
  an aspect ratio and the canvas is derived, there being only 94 legal
  canvases in the whole 1/4 to 4 range. It now states two things the old note
  got wrong by omission. The short edge is 768 only while the area cap does
  not bind, so 21:9 is 1536x672, not 1536x768. And 1.00x is not the cost
  ceiling: rounding each axis to 32 puts 29:9 (1856x576) at 1.07x, above
  16:9, for no extra pixels.

### Fixed

- `UIGraph.add` turned a dict of widget values into a list of its keys.
  Nothing had passed a dict before `VHS_VideoCombine`, whose widget set
  depends on the selected format and so serializes keyed rather than
  positionally. The graph loaded and rendered with every setting wrong.
- Both validators treated format-dependent widgets as errors, then as
  invisible. `build_workflows.validate_api` called `pix_fmt`/`crf`/... unknown
  inputs, though VHS reads them from `**kwargs` and defaults any it does not
  find. `check_workflow_schema` only understood list-shaped `widgets_values`,
  so a keyed one fell through both branches and the node went unchecked while
  reporting ok. Both now resolve the selected format's own widget list out of
  `/object_info`. Third false-positive class of this shape, all of them nodes
  whose input set is not fully static.

## 0.3.7

### Added

- `h3_text_to_video_turbo.json`, on the 8-step v1.0 LoRA at 8 steps. It
  carries a note on what the training resolution means, which is the part
  that is not obvious: 544p and 768p are about a factor of two apart in
  tokens, and H3's own canvas rule is a 768 short edge, so a 544p-distilled
  LoRA cannot be at home at the same time as the base model. Rendering at
  1344x768 keeps the base model in its trained canvas and the LoRA off its
  distillation resolution; rendering at 544p reverses that. Which costs less
  is unmeasured, and the note says so rather than implying the LoRA wins.
  t2v deliberately, because `MiniMaxH3KeyframeCanvas` refuses sub-768 and an
  i2v graph could not demonstrate the choice its own note describes.
- `steps` and `shift` are builder parameters now, so a variant can move them
  without a second copy of the graph description.

### Changed

- The `head_chunks` tooltip said "1 disables it" four lines above saying that
  1 defers to KJNodes' published count. Both were true and together they were
  misleading. It now says 1 means this node does not chunk, that nothing
  overrides a published count, and that 1 is not the lowest-peak setting:
  4 groups peaks at 2645 MiB against 2862 at 1, because chunking rules out
  the clone that only pays unchunked.

## 0.3.6

### Added

- `MiniMaxH3SigmaShift` (ModelSamplingMiniMaxH3) in all four shipped
  workflows, at the base checkpoint's own 12/3 so it changes nothing by
  default. It is there to be edited: the turbo LoRAs inherit the sampler's
  shift rather than carrying their own, and the 4-step v1.0 768p one was
  distilled at video shift 6. That is the variant whose training resolution
  matches our 1344x768 canvas, so it is the one people will reach for, and a
  graph with no shift node gives them nowhere to set it and no hint they
  needed to. Steps move with the LoRA too: 16 is a base-model number against
  these LoRAs' 4 or 8. Specs in `h3_config.SIGMA_SHIFT`, sourced from
  `coderef/Minimax-H3-Turbo`.
- The shifts joined the UI/API cross-check, for the same reason the
  checkpoint did: they are a free builder value the two formats can now
  disagree about, and a graph sampling off the wrong schedule renders
  cleanly rather than failing.

## 0.3.5

### Added

- A `unchunked_hands_over_ownership` case. sage-fork measured (`d59b82d`)
  that a slicing loop suppresses the release as completely at one group as at
  four -- the saving tracks whether the caller still holds the parents, not
  the group count. That makes the `n <= 1` branch's direct hand-over
  load-bearing rather than a shortcut for the trivial case, and unifying the
  two paths through `_chunked_heads` would turn -286 MiB into +572 with
  nothing visible in the output. The case drops the kernel's tensors and
  asserts the fused buffer actually goes; under that mutation it reports 0
  MiB freed of 336.

### Fixed

- The canvas sequence lengths in 0.3.1 were derived by fitting `rows*41 + 494`
  to the single known 41822, which also fits `rows*38 + 3518` -- the true
  decomposition, since the keyframe conditioning rows scale with the canvas
  just as the video rows do. The anchor was right and the three extrapolated
  canvases were 432 to 1296 tokens short. Re-measured at lengths computed
  from the model's own `PackedLayout`: the peak figures move slightly and the
  9.1% holds at every canvas. Surfaced by KJNodes' new `MiniMaxH3TokenCounter`
  (`6ab7e81`), which reads the layout from the model rather than inferring it.

## 0.3.4

### Fixed

- The clone is now off whenever the heads are chunked. `_chunked_heads` holds
  q, k and v across every group, so the kernel's per-group release frees
  nothing and the clone was a flat cost: at seq=41822, chunks=4 measured 3217
  MiB with it against 2645 without -- the clone's own 572 MiB recovered by
  nothing, and 69 MiB worse than chunking nothing and cloning nothing. Worst
  for the people least able to afford it, since KJNodes'
  `MiniMaxLowVRAMAttention` publishes `minimax_head_chunks` through
  `transformer_options` and our forward honours it with our own widget at 1.
  Introduced in 0.3.1 and shipped for the length of one session.
- The gate is code motion, not a new condition: it sits below the whole `n`
  resolution including the `transformer_options` read. Written against
  `head_chunks` where the clone used to be, it would read 1 in exactly the
  KJNodes case that motivated it -- confirmed by mutating it that way and
  watching only the options-route case go red.

### Added

- `bench/bench_minimax_attn.py` gained `--head-chunks` and
  `--chunks-via-options`, the latter delivering the value the way KJNodes
  does. Different code reaching the same `n`, and a fix verified only through
  our own argument is verified for the configuration nobody affected runs.
- A `chunked_path_does_not_clone` case covering both routes. Storage aliasing
  settles it exactly at 64 rows, so the peak numbers stay a one-off here and
  the check needs no threshold and no 8 GiB.

## 0.3.3

### Changed

- The clone is now gated on sage's own
  `sageattn_consume_prefers_cloned_v(device)` as well as the mode, and asked
  in the forward where `x.device` is real rather than at graph-patch time,
  where ComfyUI may not have loaded or cast the model yet and a multi-GPU
  box would bake an answer for the wrong card. The predicate exists because
  "does the release happen" and "should I clone" are the same question only
  until sage's fused-case peak drops below what a cloning caller can reach;
  gating on it means that flip arrives on upgrade instead of needing an edit
  here. A fork without the predicate keeps today's behaviour.
- `check_clone_v_wiring.py` gained a device-gate case. sm89 answers True, so
  a forward ignoring the predicate outright was invisible to the other three
  cases; forcing sage to disagree is the only way to see the gate from this
  arch, and it is what stops the file from grading our answer against itself
  now that the node reads the same predicate.

## 0.3.2

### Added

- `bench/check_clone_v_wiring.py`. The 9.1% peak saving depends on two
  things in different files joined by one argument in `nodes.py`, and
  dropping that argument changes no output, fails no existing check, and
  renders identical frames. Three cases: the predicate agrees with the
  kernel `build_kernel` actually returns, the node passes it down, and the
  flag really does take v out of the fused buffer. Each was shown red on its
  own mutation.

### Changed

- The `smooth_k=False` kwarg is now load-bearing for a second reason, noted
  where it is set: with `smooth_k=True` the K-mean copy eats the clone's
  saving and turns -286 MiB into +286 MiB.

## 0.3.1

### Changed

- The sage forward gives v its own storage before handing q/k/v to a kernel
  that releases them, cutting peak attention VRAM 9.1% at every canvas
  measured (-286 MiB at seq=41822, -174 MiB at seq=25406) for 0.7-1.0% more
  time. q, k and v are three views of one fused qkv buffer, so releasing q
  and k frees nothing while v still pins the allocation -- the same fix
  ComfyUI made upstream in `comfy/ldm/minimax/model.py`. Tied to
  `mode_releases_qkv(mode)`, not on outright: on the `fp16` mode, whose
  kernel holds q/k/v for the whole call, the same clone costs 571 MiB.

### Fixed

- `bench/bench_minimax_attn.py` drove the forward with `rope_freqs=None`,
  which takes the eager `q_norm`/`k_norm` branch. RMSNorm returns fresh
  tensors, so q and k were separate allocations and only v pointed into the
  fused qkv buffer -- not the aliasing the real inference path has, and
  aliasing is what the bench's peak column is about. It now builds a real
  rope table, and `--probe` reports the storage q, k and v land in so a null
  peak result can be told apart from a flag that did nothing.

## 0.3.0

### Added

- `MiniMaxH3ReferenceFit` node and `bench/check_reference_fit.py`. ComfyUI
  sizes reference images with `min(1.0, 2048/min(w,h))` where the reference
  pipeline has no clamp, so it never upscales: a 512x512 reference reaches the
  DiT with 1024 latent rows instead of 16384. The node does the resize itself,
  making the stock node's own resize a bit-identical no-op, and reports
  `latent_rows` because those rows are attended at every sampling step.
- `h3_rules.py`: the reference's input limits in one place -- trained aspect
  range, the 5-15s duration window checked *after* the frame-count snap, and
  the 17n+5 grid -- each citing where in `coderef/diffusers` it comes from.
- `head_chunks` on `MiniMaxH3SageAttention`, plus honouring the
  `minimax_head_chunks` key KJNodes publishes. Off by default; it exists so
  the head-chunk A/B could be run through our node at all.
- `bench/bench_e2e_h3.py` gained a canvas axis (`--canvases`, with `cheap`
  expanding to every ratio below the 16:9 default), a VRAM-knob axis
  (`--vram-arms`), and a sampled peak-VRAM column.
- `bench/check_solattn_correctness.py` and `bench/_sol_attn_reference.py`:
  the first independent correctness check the Sol-Attn Triton kernels have
  had. The reference is kijai/comfy-kitchen's pure-PyTorch eager
  implementation, vendored from the unmerged `sol_attn` branch (`ad9a4a8`)
  because no released wheel ships it; the registry import and the
  `comfy_kitchen::sol_attn` custom-op wrapper are stripped so importing it
  cannot collide with a future real one. Kernel agrees with the reference to
  cos 0.9995-0.9999 at T=512 and T=2048, bf16 and both INT8 paths, against
  upstream's own 0.998 bar. Includes a red control (kernel at one tau vs
  reference at 20x that tau, which must disagree) and a warning if the
  shipped tau is not actually sparsifying at the tested shape. Bounded by
  the reference being O(T^2): it refuses past 4 GiB of scores, so this
  checks the kernel's arithmetic, never its behaviour at H3's real length.

- `bench/check_workflow_schema.py`: validates saved UI workflows against a
  live `/object_info`. `build_workflows.py` only ever validated the API
  graphs, which carry no widget list and no slot table, so widget/socket
  confusion and link-table corruption were invisible to it. Calibrated by
  requiring a clean report on a graph ComfyUI itself wrote; two
  false-positive classes (widget-backed input slots, dynamic member slots)
  were found that way.
- `bench/check_lowvram_handoff.py`: asserts the sage forward accepts the
  single-item list KJNodes' low-VRAM block patch hands to attention.

### Fixed

- `head_chunks` moved after `patch_token_refiner` on `MiniMaxH3SageAttention`.
  Widget values map positionally in saved graphs, so a new widget at index 1
  landed an older graph's `patch_token_refiner=False` on an INT with `min=1`.
  The same hazard had been avoided one commit earlier on
  `MiniMaxH3KeyframeCanvas`'s output slots — the rule is that widgets are
  positional too, not just outputs. `bench/check_workflow_schema.py` caught
  this on its own, which is the first time a check added in this release went
  red on a defect that was not a deliberate mutation.
- `attention.py` no longer raises `AttributeError: 'list' object has no
  attribute 'shape'` when KJNodes' `MiniMaxLowVRAMAttention` is in the same
  graph. That node patches the *block* forward unconditionally to hand `x`
  over in a list, and only skips the *attention* forward if another patch
  owns the key -- so our forward received the list in either node order, and
  again from Sol-Attn's compose gate on calls it declines. The crash was
  outside the try/except, so it killed the render instead of degrading.
- `build_workflows.py` emitted Sol-Attn's `tau_profile` as a 13th widget
  value on a node with 12 widgets. It is declared `force_input=True`, which
  makes it a socket; the graphs now declare the socket and drop the value.
  Harmless in effect -- it sat past the end of the widget list, where
  LiteGraph drops it -- but the node carried a widget count no build of
  Sol-Attn has had.
- `build_workflows.py`'s own widget check counted `force_input` inputs as
  widgets, which is why it never saw the above. It also only flagged a
  shortfall, never a surplus.

### Changed

- `LONG_LENGTH` 362 -> 345. 362 frames is 15.083s against the reference's 15s
  ceiling, which it checks *after* the 17n+5 snap; 345 (14.375s) is the
  largest legal count. Breaks comparability with every recorded 362-frame
  measurement, and `h3_config.py` says which ones. `build_api` now refuses an
  illegal length or aspect rather than emitting the graph -- a comment did not
  stop 362 shipping for a week, since it is on the grid, inside ComfyUI's own
  3600 limit, and renders fine.
- `h3_config.py` records the head-chunk A/B it had been asking for since
  August. Head chunking frees 3227 MiB (three times the earlier estimate) and
  costs 0.2%; head and FFN chunking together use 1904 MiB *more* than baseline,
  so the two are antagonistic rather than additive. Noise floor stated: the
  baseline's own run-to-run VRAM spread is 784 MiB, which puts the FFN arm's
  +674 inside it and not evidence of anything.
- Workflows decode through `minimax_h3_video_vae_int8_convrot` instead of
  the fp16 VAE. Decode 12.8s -> 9.9s (1.29x) at 124 frames, 1344x768, and
  2300 MiB less resident. Quality measured on paired arms sharing a seed, so
  the latents are identical and every pixel difference is the decoder: 40.3 /
  41.8 dB against a different-seed control at 13.8 dB, and the temporal axis
  `h3_config.py` specifically warned about is flat (per-frame error std/mean
  0.063, frame-to-frame motion energy int8/fp16 = 0.998, where flicker reads
  above 1). Measured at 124 frames, not the 250+ this config usually runs.
- `bench_e2e_h3.py` reports `VAEDecode` time alongside sampler time, and
  `--video-vae` takes a list, crossing the VAE with `--arms` so a VAE
  comparison alternates instead of running as two invocations that share no
  thermal state. A VAE swap invalidates the sampler too, since
  `MiniMaxH3ImageToVideo` takes the same VAE for keyframe encoding.
- `attention.py` documents that it mirrors the stock forward's inference
  path only. Upstream grew a `comfy.model_management.in_training` branch
  using the non-in-place `rms_rope_split_half`; mirroring it would be
  theatre, since sageattn has no backward.
- `build_kernel` records why KJNodes' "pad V to CTA_K=128 in H3 mem-eff sage
  sm90" fix does not apply here: that bug comes from reimplementing sage's
  internals, and `sageattn_consume`'s fp8 dispatch excludes sm90.

## 0.2.0

### Added

- `workflows/h3_image_ref_plus_text_to_video_ref_lora.json` (and its `_api`
  copy): the shipped image-ref graph with one thing changed, loading fl2va
  plus Kijai's extracted ref LoRA instead of loading ref2va. Seed, prompt,
  canvas, length, sampler, sage and every Sol-Attn setting are shared with
  its sibling by construction, so anything that differs between the two
  renders is the LoRA.
- `REF_LORA` and `REF_LORA_STRENGTH` in `workflows/h3_config.py`.
- `unet`, `lora`, `out_prefix` and `title` parameters on `build_api` and
  `build_ui`. Checkpoint and task had to come apart: the new graph wants r2v
  conditioning driven by the fl2va checkpoint, which no task name describes.

### Changed

- Both workflow-emitting loops now read one `GRAPHS` table, so the
  difference between the shipped ref graph and its ref-LoRA sibling is a
  single `extra` dict and the two cannot drift apart in a dimension nobody
  intended.
- `cross_check` compares `UNETLoader.unet_name` and
  `LoraLoaderModelOnly.strength_model`, not only node presence. Until `unet`
  and `lora` became free builder parameters the checkpoint was derived from
  `task` inside both builders and the two formats could not disagree about
  it. Now they can, and the node-set check cannot see it, because both
  formats carry a `UNETLoader` either way. Both assertions were confirmed to
  fail when the formats are deliberately desynced.
- `coderef/` is ignored. It holds machine-local symlinks to sister repos
  rather than repo content.

The three existing graphs regenerate byte-identical.

## 0.1.0

Initial state, covering everything before the repo took an interest in the
ref LoRA.

### Added

- MiniMax H3 SageAttention node. Replaces each DiT block's attention
  `forward` with SageAttention's INT8-QK / FP8-PV kernel on Ada (sm89), and
  registers itself as the `optimized_attention_override` fallback so a
  sparse-attention patch layered on top composes with it instead of
  replacing it.
- MiniMax H3 Keyframe Canvas. Fits a keyframe to the target canvas before
  it reaches `MiniMaxH3ImageToVideo`, whose own resize stretches
  non-uniformly.
- MiniMax H3 Provenance Stamp, bench only. Records what a render's settings
  resolved to rather than what was typed, which `/history` already has. The
  field it exists for is `n_sparse`, the sigma window intersected with the
  sampler's schedule, which is readable from nothing else.
- Assert Sage Attention Chain. Turns a silently bypassed attention patch
  into a hard failure instead of a render that succeeds and is quietly
  slower or numerically different than intended.
- Benchmarks under `bench/`: module-level and end-to-end sage A/B,
  correctness check, keyframe-canvas check, override-routing check, and a
  smoke pass.
- Three generated workflows in `workflows/`, in UI and API format, built
  from one description so the two formats cannot describe different graphs.
- `workflows/h3_config.py` as the single source for checkpoint names,
  sampler settings, canvas geometry and Sol-Attn knobs, shared by the
  workflow generator and the bench. Both previously carried their own copy
  and those copies drifted.
- Sol-Attn interop evaluation in `docs/SOLATTN.md`, and H3 geometry notes in
  `docs/h3_geometry_and_nodes.md`.

### Changed

- Renamed from `ComfyUI-sageattn-ada` to `ComfyUI-h3-explorations`. Node
  ids were deliberately left alone: they are baked into every saved
  workflow's `type` field, and renaming one breaks every saved graph that
  uses it with no clear error in the UI.
