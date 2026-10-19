# Block-49 arms on the diner-breakup prompt, two seeds (2026-09-15)

Replication of the market-scene look (`2026-09-15_block49_market_feedback.md`)
on a second scene at two seeds, the step `docs/attention/quant_policy.md` asks for
before any row moves from "proposed" to "measured". Model: MiniMax H3
fl2va pruned int8 convrot (the shipped checkpoint). Prompt:
`prompt_bank/t2va_diner_breakup.txt`, unchanged. Graphs: the shipped
`h3_text_to_video_api.json`, `h3_probe_t2v_balanced_api.json`,
`h3_probe_t2v_exact_tail_api.json`, as committed, with only the prompt, the
seed and the output prefix patched over /prompt. Kernel: the served
`0.2.34+sol.2aff3c5`, sage v0.7.19. No timing, no capture, no observer.

Arms, in the order the owner watches them (originals, not blinded; the
owner said the blinded stacks did not help on the market scene):

| clip | arm | what differs from shipped |
|---|---|---|
| shipped | shipped chain | nothing |
| balanced | free levers | `MiniMaxH3ChannelBalance` on the loud blocks (45/48/49, from weights) + sage `fp8++ balanced` |
| exact_tail | ceiling | blocks 45/48/49 on ComfyUI's bf16 attention |

## Outputs

Under the server's output directory, `Video/block49_diner/`:

- `shipped_s730451892_00001-audio.mp4` (prompt id `943b8f77-1ff1-4143-864a-a4a9ba80e481`)
- `balanced_s730451892_00001-audio.mp4` (prompt id `c586306b-5b4b-4882-8a14-2f029d40fe6e`)
- `exact_tail_s730451892_00001-audio.mp4` (prompt id `42713e8f-62d5-4032-9b18-ed95aeb460df`)
- `shipped_s20260915_00001-audio.mp4` (prompt id `bab9465c-8a2e-4070-be24-6a9ada920f69`)
- `balanced_s20260915_00001-audio.mp4` (prompt id `00427e14-406d-4ff5-8ac7-ebc84717aede`)
- `exact_tail_s20260915_00001-audio.mp4` (prompt id `5b926246-0e88-4b12-a2a0-3dde57137534`)

Two seeds: `730451892` and `20260915`. Same seed across arms is not the
same take: any numerics change moves the trajectory, so what is compared is
the reading of the prompt, the identity holding, and the physical
consistency of objects, not frame-level agreement.

## What to look for (the market scene's separators)

- Identity: does either character change face or clothing mid-clip.
- Physical consistency of props: the market morph was the crate never being
  decided as one deep box or two shallow trays; here the diner table's
  props (cups, plates, the check) are the analogue.
- Prompt reading: which actions and cuts the prompt names actually happen.
- Audio: the ceiling arm sounded louder and crisper on the market seed; is
  that a scene effect or a numerics effect.

## Owner's scoring

Seed 730451892, originals watched in listed order (clip 1 = shipped,
clip 2 = balanced, clip 3 = exact_tail), the owner and other viewers:

> everyone says clip 3 is best, 2 is second best, 1 has some issues at the
> end ("The only thing that immediately stood out is a chef morphs out of
> thin air at the end of the first one").
>
> as i watched, 3 is just a much better scene in subtle ways. even 2 is
> better than 1 - it has text on the sign in the background that isnt
> illegible. in clip 1, you can immediately see the sign on the door says
> blue star diner twice. which is weird. actually it says "TUE SATR DINER"
> in clip 1 on the door.

Same ranking as the market scene (shipped < balanced < exact tail), on a
second scene and a second seed; the shipped arm's failure is again a
morph (a chef appearing from nothing at the end) plus a doubled,
misspelled sign. Seed 20260915: pending.

## Wall time per render (server history, execution_start to execution_success)

Same server, same card, one render at a time, the diner batch back to back
and the market three earlier the same day; includes VAE decode and audio.
Seconds, rounded.

| arm | diner seed 730451892 | diner seed 20260915 | market (2026-09-15, earlier) |
|---|---|---|---|
| shipped | 480 | 478 | 508 |
| balanced | 479 | 479 | 491 |
| exact_tail | 557 | 557 | 567 |

The two free levers cost nothing measurable (a weights fold at load and a
per-call factor in sage's quantizer). Exact attention on three of fifty
blocks costs about a sixth more wall time: bf16 attention at S~104k on
those blocks runs without Sol's routing and without INT8.

## Audio loudness per clip (ffmpeg ebur128, integrated LUFS / true peak dBFS)

| arm | seed 730451892 | seed 20260915 |
|---|---|---|
| shipped | -28.8 / -13.2 | -28.4 / -12.1 |
| balanced | -28.8 / -11.8 | -27.3 / -11.3 |
| exact_tail | -29.3 / -12.5 | -27.6 / -10.0 |

No arm effect: all six within about two LU. The market scene's louder
ceiling arm was that take, not the tail.

## Stacks

Captioned vertical stacks (video only, `bench/stack_labeled_clips.py`), one
per seed: `Video/block49_diner/stack_diner_s<seed>_default_rebalanced_bf16tail.mp4`.
Band order: default (no rebalance), rebalanced in its earlier form (node +
sage `qk_balance`; Sol's own factor did not exist yet when these
rendered), bf16 on blocks 45/48/49 with no rebalance. Note the arms are
NOT the market and kitchen stacks' arms: there the middle band has Sol
balanced too and the third band is rebalanced plus the bf16 tail.
