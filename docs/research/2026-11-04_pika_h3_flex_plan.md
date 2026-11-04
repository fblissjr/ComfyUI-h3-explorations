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
- **Reference path:** core's `MiniMaxH3ReferenceToVideo` first, at 1344x768 and 124 frames, which is what
  Comfy's Pika template uses. The pack's typed conditioner comes second, as its own arm. The repo already has
  probe arms comparing the two paths (`workflows/INDEX.md`, `h3_probe_ref_pathway_*`); this plan has not read
  their results.
- **Attention:** dense to begin with, matching Pika and Comfy, so step 2 tests the pipeline and not Sol. These
  graphs need an entry in `SOL_EXEMPT_STEMS` until step 4 settles it, and the reason has to be written there.
- Decides nothing; makes steps 2 and 3 possible. Likely to work: every part is an existing node.

### 2. Does this stack reproduce Pika? (about 8 renders)

- **Inputs:** Pika's own showcase material (`assets/showcases/` in the early-access download), all at
  1344x768. `motion-transfer` (a video plus two images) is the reference case; `balcony-orbit` is text only.
  Pika's manifest says seeds and preprocessing are not given, so this is a quality comparison by eye, not a
  match. Two seeds each.
- **A second arm at no extra design cost:** the same seeds through `LoraLoaderModelOnly`, the template's merge
  path. It settles whether the merge noise shows on screen.
- **Decides:** whether to go on. If our renders look clearly worse than Pika's, the fault is in the stack (the
  reference path, preprocessing, prompt form) and every later step waits.
- **Likely:** text-to-video works; this is Comfy-Org's own graph with the branch swapped in. Reference-to-video is
  the real risk. Pika's service does its own preprocessing, and the prompt refers to references by tag
  (`@Image1`), so the reference path and prompt form are where a gap would come from.

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

### 5. One activation run (no render, one server run)

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

- **Arms**, all files the converter already makes, against full rank:
  - spike only and tail only;
  - switch strength 0.9 with body 1.0;
  - `svd99`;
  - kijai's file, as the control;
  - task boost, β about 0.25, on ref2va.
- Two shots, a repeat. Order the arms by step 5's ranking.
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

- **Make the measurements one tracked tool.** The spectrum map, the fl2va/ref2va split, the truncation
  projection and the AdaLN alignment, so §7's figures point at a command (the repo's rule).
- **Records.** Each render step gets a record under `bench/results/` (its `.md`), data under `data/`, and video
  under the share's `output/Video/`.
