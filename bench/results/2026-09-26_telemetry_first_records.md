# Pipeline telemetry, first records: the shipped FlashGen t2v graph, 2026-09-26

**What ran.** `h3_text_to_video_flashgen` at seed 730451892, 1344x768, 345
frames. The server was a fresh process started with `H3_TELEMETRY`
(`docs/pipeline_telemetry.md`, schema 1, 250 ms sampler, block residency on).
Two renders, one after the other: a cold warmup, then a warm render whose
conditioning and loaders came from cache.

- Timing rows: `2026-09-26_telemetry_overhead_s1.jsonl`.
- Per-node summaries from `bench/telemetry_report.py`:
  `2026-09-26_telemetry_flashgen_cold.json` and `_warm.json`.
- The raw records stay under the capture root (`2026-09-26_telemetry/`) and
  are not committed. Their headers carry the machine's paths, and the
  summaries redact them.

## What arming costs

The warm armed render took 165.9 s in total (sampler 131.6 s, decode 29.9 s).
The warm unarmed render of the same graph and seed earlier that day
(`2026-09-26_draft_decode_s1.jsonl`, row `ship`) took 166.1 s (131.7 s,
29.9 s). The difference is inside run-to-run noise: arming costs nothing
measurable at this interval. The two ran in different server processes.

## What the records show

Numbers are from the two summaries. The prose says what they mean.

- **The DiT never fits, and dynamic VRAM keeps about half of it on the
  card.** After sampling, `MiniMaxH3` has about 10 GB of its 20 GB resident,
  and every block is at the same resident fraction. The other half streams in
  from pinned host memory on each step. During the warm sampler PCIe carried
  50.8 GiB to the card, a mean of 0.39 GiB/s with bursts to 12.4 GiB/s. The
  card sat at 99.5% mean utilisation, so the streaming overlaps compute rather
  than stalling it.
- **The link is PCIe gen 4 x8.** The card's maximum is x16 (header
  `pcie_width_max`, sample `pcie_width`). The bursts reach roughly what x8
  gen 4 can carry. `docs/hardware.md` already notes the narrowed link.
  Whether x16 would shorten a render is not answered here: the mean transfer
  rate is far below the link and the GPU is busy.
- **The text encoder is evicted by the sampler and the video VAE by the
  sampler, then each comes back.** After conditioning, the encoder
  (`MiniMaxH3TEModel_`, 25.9 GB) holds about 19 GB on the card. By the end of
  sampling it holds almost nothing, with 5.2 GB of it pinned in host RAM. The
  video VAE, fully resident after a decode, is at zero by the end of the next
  render's sampler. It is reloaded inside VAEDecode: about 4.9 GiB crossed to
  the card in that window, and the decode took the same time as the isolated
  decode in `2026-09-26_vae_decoders_345f.md`. So the reload is not where the
  decode's time goes.
- **Host RAM.** Cold, the sampler's first use of the DiT added about 8.8 GB
  of anonymous memory to the server process (the pinned DiT, `pinned` 8248
  MiB), and the page cache shrank by a similar amount. The warm render moved
  neither. VAEDecode adds about 4 GB of anonymous memory, the size of the
  decoded frames in float32. The weights came from page cache, not disk: the
  cold sampler read 1 MiB from disk with 11 major faults, and conditioning
  read 406 MiB.
- **torch's allocator does not see dynamic VRAM.** At the sampler's end,
  torch reports about 11 MiB allocated and 64 MiB reserved, while dynamic
  VRAM (`aimdo_vram`) and the models' own accounting both put about 10 GB on
  the card. Read `aimdo_vram` or NVML's `used`, never torch's figures. The
  schema doc inferred this before the record; it is now measured.
- **Peak card use** during sampling was about 24.3 GB by NVML, which is the
  whole card. Dynamic VRAM fills what it is given, so a peak here says
  nothing about headroom. The per-model residency is the figure that does.

## Reader note found on this record

A node can log `node_start` more than once. Core looks a node up when it
first reaches it, and again when it runs after waiting for its inputs. The
reader pairs the last start before the end, which is the execution. A
sample's `node` field can name the waiting node for the moment between the
two.
