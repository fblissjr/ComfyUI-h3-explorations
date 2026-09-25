# PDD upstream: can we do better than core's and our own?

last updated: 2026-09-25

> **Provenance, 2026-09-25.** A research subagent (Claude) read the seven
> implementations and wrote the probes. The session that commissioned it
> re-ran every probe from `bench/`, checked the load-bearing claims against
> source and the Hugging Face API, and wrote this note. The owner reopened PDD
> for this research on 2026-09-25 (`../../wiki/decisions.md`). A render in
> this lane still waits on the owner's go. The numbers are in
> `../../../bench/results/2026-09-25_upstream_pdd_comparison.md`; this note
> carries none of them.

**The answer.** No upstream implementation beats ours on anything that
reaches a shipped render. Ours sits at the fp32 floor for fused-head
precision, as do the vendor, UtilsCollection, and core when fed correct
deltas. Ours and core pick the same heads on every shipped schedule without
an audio mask. The places where ours could still improve are listed under
"Arms", and each needs the owner's go for a render. The adopt-upstream rule
does not fire on any axis: sglang and core agree with ours on eight
evaluations, shifts and Euler, and disagree with each other on everything
else.

**Two findings change what the repo believes:**

1. **Core's audio carry is exact under Euler, so the "change of variable"
   mechanism for PDD's audio loss is refuted.** Given the same velocity,
   core's single-schedule carry lands exactly where the vendor's two-schedule
   audio step lands, at every partition tried. The algebra: the carry factor
   is affine in σv, so the carried audio is a straight flow path, and the
   velocity transform at the block start is the chain rule along it. Scope:
   Euler, scheduled sigmas, unmasked audio. What this means:
   - PDD's audio energy loss at coarse partitions is PDD's own, and the vendor
     has it too.
   - "Vary the transform at fixed partition" has no mechanism left to find.
   - The 2026-08-28 mean-sigma ablation's energy gain is a departure from the
     vendor's integrator, which happens to add energy, not a fix.
   Corrected in place, with dated notes, in `audio_under_pdd.md`,
   `../../h3_pdd.md` and `../../evidence.md`.
2. **The Kijai PDD files symlinked into this install are stale, and merged
   core decodes them to a roughly doubled head.** They predate Kijai's
   2026-08-27 re-upload, which moved the banks to deltas to match core's
   merged formula. No shipped graph names them. Re-fetch or unlink them: the
   owner's call.

## The seven implementations

| axis | vendor (alibaba-pai) | sglang | core (merged `2504e68d`) | ours | UtilsCollection | T8 | LongMedia |
|---|---|---|---|---|---|---|---|
| partition | uniform, sigma points = evaluations + 1 | uniform, fixed when fused | any schedule, from `sample_sigmas` | any; knots from `sample_sigmas`; emits SIGMAS; ManualSigmas tail6 ships | explicit widths of 4 or 8 | the official 8-step schedule only | as core |
| `denoise` < 1 | none | refused | works | works | only on its partition | refused | as core |
| head selection | forward-hook counter (safe only because the model is guidance-distilled) | loop step counter | sigma argmin, inverse shift; one block for both streams | each stream's own `t_emb` row against boundary embeddings | sigma from the timestep, knot tolerance | nearest official sigma | as core |
| head precision | fp32 heads and plan (diffusers keeps `proj_out` fp32) | bf16 plan, bf16 storage | fp32, einsum each forward | fp64 fuse, fp32 cached per block | fp64 then fp32 | bf16 deltas, or a bf16 fallback | as core |
| backbone | runtime LoRA on unpruned bf16 | offline merge into unpruned bf16 (fc1 half-swap defect: `../../../bench/results/2026-09-25_sglang_pdd_fc1_merge.md`) | whatever the file carries | runtime, unmerged, or the INT8 bake; pruned or unpruned | runtime | unpruned only | none |
| strength | none | none | LoRA strength | backbone and head separately | backbone and head | 0 to 1, head lerp below 1 | n/a |
| audio | own scheduler at shift 3 | own schedule | carry (exact, finding 1); audio block = video block | carry; audio block from its own row | refuses other shifts | dual-clock Euler (equivalent); optional audio refine on the base model | as core |
| guards | none | grid and shift refusal | only a missing `sample_sigmas` | partition fingerprint, shift raise, tiling, patch clash, bake pairing; off-schedule warns | envelope, off-grid, shift | exact schedule, unpruned, strength range | as core |
| licence | Apache-2.0 | Apache-2.0 | GPL-3.0 | MIT | AGPL-3.0 | GPL-3.0-or-later | Apache-2.0 |

Pointers:
- vendor: `coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs/minimax_h3_pdd.py`;
- sglang: `coderef/sglang/python/sglang/multimodal_gen/tools/fuse_minimax_h3_pdd_heads.py`;
- core: ComfyUI checkout `comfy/ldm/minimax/model.py`, `FinalLayer.forward`;
- ours: `pdd_lora.py`, `pdd_math.py`;
- UtilsCollection: `coderef/ComfyUI-UtilsCollection/helpers/patcher_helpers.py`;
- T8: `coderef/comfyui-minimax-h3-audio-T8/h3_t8/pdd_advanced.py`;
- LongMedia: `coderef/ComfyUI-MiniMax-H3-LongMedia/nodes.py`.

Nothing else implements PDD. vllm-omni's "pd" code is prefill/decode
disaggregation, and LightX2V, DiffSynth-Studio and diffusers have none.
UtilsCollection's pinned version patches `final_layer` with a four-argument
forward, which merged core calls with seven. It is not installed here.

## Evidence for the open decision: does our node keep its head half?

`../../h3_pdd.md`, "Core is learning this", left this open when core's PR
merged on 2026-08-29.

**Precision is not an argument either way.** Core with correct deltas sits
at our floor.

**What favours keeping ours:**
- **Core never engages for our files.** Our converted and stripped files
  carry no enlarged bank, and the baked checkpoint's heads are single, so core
  takes its one-head branch and our object patches run the head in every
  shipped graph that patches heads. Deferring means switching to Kijai's current
  delta-encoded file, which sits at our precision under core, or re-encoding ours
  to match.
- **Core has no partition fingerprint and no shift guard.** It raises only
  when `sample_sigmas` is missing.
- **Core picks blocks the audio never visits on masked rows.** On the three
  shipped song graphs (`audio_mask` 0.25), core gives the pinned audio row
  the video's blocks. Ours picks from the row's own clock. Neither choice is
  trained, and the visible effect is unmeasured.
- **Core keeps the enlarged fp32 banks resident on the card**, where ours
  holds one fused head per stream.

**What favours core:**
- One mechanism instead of two in the process.
- A shared, shift-invariant index that stays in step across streams on
  schedules finer than the grid, where ours can drift by one interval
  (synthetic only).

Unverified: whether the 2026-08-27 lingering-enlarged-weight crash still
reproduces now that core checks shape-changing patches.

## Arms, if the owner wants more from PDD

Each needs a render or a capture; none has been run.

1. **Selector rule for mask-pinned audio rows** (no upstream has this).
   - Keep a shared grid index taken from video.
   - Use the audio row's own clock only when a mask pins it.
   - Stop the off-schedule warning from firing on such rows. On the three
     song graphs it would fire once per render (inferred from the probe; no retained server log covers a song-graph run).
   - Instrument: a capture of the audio head's output on one song-graph
     window, against the fp64 mean head over the row's actual span.
   - Reach: the song graphs only.
2. **Audio refine on the base model after PDD** (T8 implements it for its
   `pdd8` and `pdd4_plus4` modes; proposed here in `audio_under_pdd.md`
   section 3.4 and never run).
   - Its old rationale, the carry, is gone.
   - The practical one stands: coarse partitions lose audio energy.
   - Instrument: a blind pair through the repo's process.
3. **PDD on the unpruned base.** sglang and T8 both run it unpruned. No
   rendered unpruned PDD arm was found in the records here, though that was
   not searched exhaustively.
   - Instrument: a capture-graded first-step velocity, then a blind pair.
4. **T8's 4+4 two-pass.** A speed arm, not a quality one; large.

**Not arms:**
- precision (we are at the floor);
- a dual-clock sampler (equivalent to core's carry under Euler, so a no-op).
  Re-applying the carry at the block's mean sigma is different. It is not a
  correction, but it does add audio energy
  (`../../../bench/results/2026-08-28_audio_carry_ablation.json`). That makes it
  an untrained gain knob, measured and never judged by ear. If louder PDD audio
  is the goal, arm 2 is the principled route, and this knob is a cheaper one to
  put beside it in the same pair;
- an off-grid refusal (every shipped schedule is exact);
- partition arms, which the owner ruled out on 2026-09-05
  (`../../roadmap.md`, "Owner decisions, 2026-09-05 evening").
