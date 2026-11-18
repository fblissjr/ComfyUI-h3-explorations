# `--fast fp16_accumulation` and the H3 video VAE on the RTX 4090 (2026-11-18)

**Question (from the dotfiles session, with the owner's approval).** In ComfyUI 0.45.0 with comfy-kitchen
0.2.41+sol, `--fast fp16_accumulation` sends the H3 video VAE's fp16 work to kitchen's fp16-accumulate kernels.
Comfy's 2026-09-22 post measured that on an RTX 5090. What does it do on this card, in time and in fidelity, and is
it worth passing per launch?

**Answer, on 141 frames at 1344x768:**

| | flag off | flag on | change |
|---|---|---|---|
| encode, either file | 19.3 s | 13.2 s | 32% faster; encode error from fp32 up 31% |
| decode, `minimax_h3_video_vae_int8_convrot` (shipped) | 7.45 s | 7.42 s | no change in time; 1.1 dB further from fp32 |
| decode, `minimax_h3_video_vae_fp16` | 17.7 s | 12.7 s | 28% faster; 5.5 dB further from fp32 |

The encoder half is the same fp16 network in both files. The int8 file quantizes only the decoder's linears.
`2026-09-27_vae_encoder_fp16acc.md` found "no change" on one random frame. That finding does not describe a
clip encode on today's core and kitchen.

**Where a shipped graph pays the encode:** only where it encodes a clip: `<Video N>` references, masked_v2v
sources, guide clips of 5 frames or more. None of the 14 graphs with `MiniMaxH3AppendRefVideo` or the 8 with
`MiniMaxH3MaskedSource` sets `MiniMaxH3VAEPrecision`, so they encode in fp16, where the flag reaches. Text to video encodes nothing. Single-frame encodes (stills,
`<Picture N>`, the image graphs) skip most of the kernel (section 3). The image edit graphs encode in fp32
(`MiniMaxH3VAEPrecision`), which the flag cannot reach.

**What else the flag changes.** The switch is process-wide (`comfy/model_management.py`, where it reads
`args.fast`). The 32B text encoder also takes it, since core runs it in fp16. So does any other fp16 model the
server loads, and `PRIORITIZE_FP16` moves any model that lists fp16 as supported to fp16. The H3 DiTs list only
bf16 and fp32 and are not moved.

## Conditions

- **Card:** RTX 4090, PCIe 4.0 x16, 360 W cap.
- **Software:** ComfyUI 0.45.0 (core `0263e4880`), comfy-kitchen `0.2.41+sol.b63d8c7.up.7d6a22a`, torch
  2.15.0+cu134.
- **Isolation:** run in standalone processes while the server held only its idle 500 MiB (`POST /free` first).

## Method

A throwaway script (session scratchpad, not kept) built on `bench/compare_vae_decoders.py`'s design:
- **One fresh process per arm**, so the flag reaches core through its argument parser as a server launch would.
- **Each arm** decodes one saved latent twice and encodes one real clip twice; the tables use the second call.
- **The latent:** the video half of a Pika text-to-video render at 141 frames, 1344x768 (the warm-up run of
  `2026-11-18_pcie_x16_render_times.md`, saved by `LTXVSeparateAVLatent` and `SaveLatent` to
  `$H3_OUTPUT_DIR/latents/bench_2026-11-18_link_x16/`).
- **The clip:** that record's `pika_r1` MP4 under `$H3_OUTPUT_DIR/Video/bench_2026-11-18_link_x16/`, 141 frames
  at 24 fps.
- **The reference:** the fp16 file cast to fp32 for both halves (`MiniMaxH3VAEPrecision`, `fp32`/`fp32`). It is
  the higher-precision run of the same weights, not ground truth.

Each child process asserts that torch's `allow_fp16_accumulation` matches the arm. A first pass failed silently:
`bench/_lib::bootstrap` imports `folder_paths`, which makes core parse its arguments, before the script set
`--fast`. Every "on" arm then ran with the switch off and matched "off" bit for bit. The assertion was added and
that pass discarded.

Data: `data/results/2026-11-18_vae_fp16acc_ab.json`.

| arm | decode, s (first, second) | encode, s (first, second) | decode PSNR from fp32 | encode mean abs from fp32 (max) |
|---|---|---|---|---|
| fp16 file, off | 18.26, 17.74 | 19.32, 19.31 | 62.69 dB | 0.001952 (0.818) |
| fp16 file, off, repeat | 18.38, 17.80 | 19.35, 19.33 | 62.69 dB | 0.001952 (0.818) |
| fp16 file, on | 13.35, 12.73 | 13.17, 13.18 | 57.24 dB | 0.002559 (0.976) |
| fp16 file, on, repeat | 13.32, 12.72 | 13.16, 13.17 | 57.24 dB | 0.002559 (0.976) |
| fp32 (reference) | 70.31, 69.28 | 49.79, 49.86 | | |
| int8 file, off | 7.95, 7.45 | 19.38, 19.35 | 57.48 dB | 0.001952 (0.818) |
| int8 file, on | 7.94, 7.42 | 13.17, 13.19 | 56.40 dB | 0.002559 (0.976) |

**Controls.**
- **Determinism:** the repeats are bit-identical, decode and encode, with the flag off and on.
- **Shared encoder:** the int8 file's encode equals the fp16 file's to the bit in each state, as expected.
- **On against off, directly:** decode 57.40 dB (fp16 file) and 56.60 dB (int8 file); encode mean abs 0.00167,
  max 1.005.

**Peak memory** (`torch.cuda.max_memory_allocated`) is the same with the flag off and on:
- decode: 5,650 MiB (fp16 file), 3,292 MiB (int8 file);
- encode: 6,381 MiB (fp16 file), 4,081 MiB (int8 file).

## 2. The int8 decode moves without getting faster

Its 144 quantized linears take core's int8 path, where the flag is never consulted (`comfy/ops.py`, the fp16 linear
is reached only for weights that are not `QuantizedTensor`). Some smaller fp16 work in the decoder must take it,
since the output moves. Candidates are the file's two unquantized linears and any other fp16 GEMM in the decoder.
Which one accounts for the 1.1 dB was not traced.

## 3. The pack's grader, and why it says less than the clip does

`bench/grade_vae_encoder_precision.py --fp16-accumulation`, as the request named. One random frame, the shipped
int8 file. Data: `data/results/2026-11-18_vae_encoder_fp16acc_672.json`.

| size | fp16-accumulate against plain fp16 (mean, max) | from fp32: accumulate, plain fp16 | controls |
|---|---|---|---|
| 672x384 | 0.00038369, 0.076707 | 0.00073421, 0.00072339 | determinism and direction hold |
| 1344x768 | out of memory on the card in the fp32 arm, after the determinism arm passed | | |

- **Unchanged since 2026-09-27.** The 672x384 accumulate delta matches `2026-09-27_vae_encoder_fp16acc.md` to
  every printed digit, across two kitchen versions.
- **One frame shows about 1.5% more error from fp32; the 141-frame clip shows 31%.** In `comfy/ldm/minimax/vae.py`,
  `CausalConv3d.forward` sends a single-frame input with temporal padding to `super().forward(x,
  autopad="causal_zero")`. Only the channels-last branch below it calls `_fp16_accum_conv`. So on one frame only
  the convs without temporal padding reach the kernel. A clip's 3x3x3 convs all do.
- **The tool's own claim is stale.** Its `--size` help says #33 read 1344x768, but the #33 records are 672x384. At
  1344x768 the process held 22.6 GiB before the fp32 arm failed, while the clip script's fresh-process fp32 encode
  of 141 such frames peaked at 13.4 GiB. So the tool's arms likely accumulate memory in one process; not traced. *Traced the same day: it was autograd, not the arms (section 3b).*

## 3b. The grader after its fix (same day)

`bench/grade_vae_encoder_precision.py` now encodes under `torch.inference_mode()`, as the server runs every node,
and takes `--frames` (default 22) and `--clip`. The 1344x768 failure above was autograd: the tool called
`vae.encode` in grad mode, which kept every activation. Under inference mode every arm fits; the records hold each
arm's peak (`memory`). The one-frame control at 672x384 reproduces the section 3 figures to every printed digit, so
the fix moved memory, not numbers.

The clip rows take the first 22 frames of the `pika_r1` MP4 (Method), scaled to the size. Mean absolute difference
from the fp32 encode of the same input:

| input | fp16-accumulate | plain fp16 | data |
|---|---|---|---|
| one random frame, 672x384 | 0.00073421 | 0.00072339 | `2026-11-18_vae_encoder_fp16acc_1f_672x384_inference_mode.json` |
| 22 random frames, 672x384 (the new default) | 0.00196970 | 0.00062961 | `2026-11-18_vae_encoder_fp16acc_rand22_672x384.json` |
| 22 clip frames, 672x384 | 0.00171269 | 0.00106352 | `2026-11-18_vae_encoder_fp16acc_clip22_672x384.json` |
| 22 clip frames, 1344x768 | 0.00217210 | 0.00161726 | `2026-11-18_vae_encoder_fp16acc_clip22_1344x768.json` |
| 141 clip frames, 1344x768 (Method's table) | 0.002559 | 0.001952 | `2026-11-18_vae_fp16acc_ab.json` |

All under `data/results/`. Both controls held in every run. Random pixels widen the gap well past what picture
content shows; `--clip` is the input to read a size from.

## 4. What this cannot see

- **Whether 31% more encode error matters to a render.** A render cannot A/B a numerical knob, because the
  trajectory diverges from any perturbation. The DiT-side question is the one `bench/measure_encoder_quant_dit.py`
  asks for the text encoder; nothing like it exists for the VAE encode.
- **One clip, one latent, one machine.** The content is a Pika text-to-video scene.
- **The text encoder's share of the flag.** That needs a server launched with the flag, which needs the owner's
  go; not run.
- **How this compares with the 5090 post.** The post was not read for this record.
