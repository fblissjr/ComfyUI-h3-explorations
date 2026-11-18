# The 32B text encoder, int8 ConvRot against bf16 pruned: speed at x16 (2026-11-18)

**Question (from the dotfiles session, with the owner's approval).** Every graph has loaded the bf16 pruned encoder
since 2026-11-14 (`docs/wiki/decisions.md`, owner). It is 48.0 GiB, so it cannot sit on the card next to the DiT
and crosses the link on every pass. The int8 ConvRot file is 25.3 GiB. How much time does it save, and what is
already known about its quality?

**Answer.**
- **Warm passes (the encoder file in page cache):** the int8 pass is 1.1 to 1.7 s shorter.
  - Pass times, bf16 against int8: 4.43 against 3.25 to 3.36 s (text to image), 4.31 to 4.52 against 3.16 to
    3.24 s (edit), 4.43 to 4.84 against 3.13 to 3.31 s (Pika).
  - Whole renders finish 0.8 to 1.8 s sooner.
  - That is about the size of the x8-to-x16 gain on the same graphs (`2026-11-18_pcie_x16_render_times.md`), not
    larger.
- **Passes that read the file from the drive:** int8 is roughly twice as fast, because there is half as much to
  read. bf16 took 16.6 to 21.5 s against 9.5 to 11.4 s for int8. The edit graph's reference encode took 24.2 s
  against 11.0 s.
- **Sampling, decode and the DiT's return** take the same time with either encoder.
- **Quality is already measured; see section 3.** In short: text tokens stay close, while image tokens carry a
  heavy error tail, and no DiT-side test on the int8 DiTs can resolve it. Choosing the default stays the owner's
  decision.

## Conditions

Same server, stack and card as `2026-11-18_pcie_x16_render_times.md`:
- **Card and server:** PCIe 4.0 x16, 360 W cap, ComfyUI 0.45.0 with comfy-kitchen
  `0.2.41+sol.b63d8c7.up.7d6a22a`, no `--fast`, dynamic VRAM on.
- **Run time:** 12:23 to 12:37 Central, right after the VAE runs of `2026-11-18_vae_fp16_accumulation.md`, which
  loaded seven VAE processes and may have pushed model files out of page cache.
- **The encoder computes in fp16 either way.** Core's `text_encoder_dtype` returns fp16, and the server log reads
  `CLIP/text encoder model load ... dtype: torch.float16`.

## Method

A throwaway harness (session scratchpad, not kept):
- **Graphs and prompts:** the same three graphs and appended-sentence scheme as the link record, with new
  sentences so nothing was a cache hit.
- **The only change** was `MiniMaxH3EncoderLoader.encoder_name`.
- **Order per graph:** bf16 warm-up and two runs, then int8 warm-up and two runs.
- **Timing:** each node's time is the interval between the server's WebSocket `executing` events, so the encoder
  pass is the conditioning node's own time: `MiniMaxH3ImageConditioning`, `MiniMaxH3PromptOnImageReferences`
  (edit) or `MiniMaxH3Conditioning` (Pika). The edit graph's `MiniMaxH3EncodeImageReferences` (the picture's
  vision pass and KV prefix) runs only on a warm-up; its output is cached after that, since the picture does not
  change.
- **Link traffic:** `nvidia-smi dmon -s puct -d 1` throughout.

Data: `data/results/2026-11-18_text_encoder_int8_vs_bf16_speed.json`.

## 1. Times

Seconds; warm-up, run 1, run 2.

| graph | encoder | encoder node | sampler node | whole render |
|---|---|---|---|---|
| text to image | bf16 | 16.68, 16.56, 4.43 | 17.47, 17.43, 17.48 | 38.54, 40.56, 25.75 |
| | int8 | 11.37, 3.36, 3.25 | 17.46, 17.47, 17.47 | 33.53, 24.65, 24.58 |
| edit | bf16 | 2.46 (+24.23 reference), 4.31, 4.52 | 20.90, 20.95, 20.98 | 51.58, 28.72, 28.97 |
| | int8 | 1.70 (+10.97 reference), 3.16, 3.24 | 20.93, 20.95, 21.41 | 37.66, 27.56, 27.88 |
| Pika, 141 frames | bf16 | 21.49, 4.84, 4.43 | 56.00, 52.41, 52.36 | 88.50, 66.66, 66.37 |
| | int8 | 9.53, 3.31, 3.13 | 52.27, 52.20, 52.30 | 72.02, 64.90, 64.96 |

**The bf16 text-to-image run 1 was a second slow pass**, at 16.56 s, before run 2 ran at 4.43 s.

## 2. What the link carried during the encoder node

`dmon`'s receive rate, one sample a second (a rate at an instant, not a byte count):
- **Fast passes, either file:** peaks of 14 to 22 GB/s, with near-zero seconds between them. Both encoders stream, because the DiT
  holds the card. The int8 file has half as many bytes to move.
- **Slow passes, either file:** a long plateau at 3.3 to 3.8 GB/s with gaps. That looks like reads from the model
  drive (PCIe 4.0 x4 NVMe) rather than from RAM. So a slow pass is a page-cache miss, as
  `2026-11-17_h3_encoder_streaming.md` reasoned for its 17 s pass.
- **Residency was not checked.** `fincore` is not installed, so how much of each file sat in page cache was not
  measured. The machine showed 94 GiB of buff/cache afterwards.

**A correction to the premise.** The request read "loaded partially; ~21 GB on the card, ~28 GB offloaded" from the
x8 server log as the bf16 encoder's per-pass transfer. In that log those lines appear only in the `keep_in_ram`
arm, which used core's classic patcher (pid 291927, from 13:50 Central on 2026-11-17). Under dynamic VRAM, the
default, the server prints no such line on any run. So the volume streamed per pass comes from `dmon`'s rates, not
the log, and is approximate.

## 3. Quality: already recorded, not re-derived

- **Conditioning, int8 against bf16 pruned, on shipped graphs and the violin-maker scene in three modes**, dated
  2026-09-27 (`docs/wiki/decisions/2026-09-25_to_2026-09-29.md`):
  - the vision tower and embedding table are bf16 in both files and add no error;
  - text tokens stay close;
  - image tokens carry a heavy tail that the int8 decoder layers add, and one violin ref2va patch is inflated well
    past its bf16 norm.

  The data is `data/results/2026-08-18_to_2026-10-03/2026-09-27_encoder_int8_vs_bf16_conditioning.json`. The
  path cited in `workflows/h3_config.py`, `docs/wiki/stages.md` and `bench/measure_encoder_quant_dit.py` lacks
  the dated folder. Most `data/results` citations in tracked files have the same problem since the archive move.
- **Whether that reaches the DiT** is open experiment #47 (`docs/wiki/ops/open_experiments.md`), closed unbuilt on
  2026-09-27. Noise far below int8's error moves the int8 DiT's prediction as much as int8 does, so no per-step
  test on these DiTs can resolve encoder precision.
- **Still true on today's kitchen.** The 2026-11-11 kitchen rebuild left the int8 encoder's conditioning bit for
  bit unchanged, because its MLP is too wide for the changed fused path
  (`2026-11-11_kitchen_upstream_sync_and_two_fixes.md`).
- **Fixed-seed renders were not made.** A render cannot A/B a numerical change: the two arms would be different
  samples, not better and worse versions of one.

## 4. What this cannot see

- **Page-cache state per pass.** The cold-pass figures depend on what else the machine loaded, here a VAE study
  just before. On a box that keeps the 48 GiB file cached, only the warm figures apply.
- **One machine, one seed per graph, two measured runs per arm.**
- **Encoder pass time under `--fast fp16_accumulation`.** The flag reaches the encoder's fp16 linears (bf16 file)
  but not the int8 file's quantized ones. Not run, because it needs a server launch with the flag.
