# The H3 text encoder's pass time, and `MiniMaxH3EncoderLoader.keep_in_ram`, end to end (2026-11-17)

**Question (the owner's).** In a 16 MP MiniMax-H3-Image text-to-image render, the Qwen3-VL-32B encoder pass took about
as long as all of sampling. Why, and does loading the encoder with core's classic patcher
(`keep_in_ram`, merged as opt-in the same day) make renders faster?

**Answer.** No. On the running server, `keep_in_ram` made every measured render about 12 s slower end to end.
- **The encoder pass itself was no faster:** about 7 s either way, with the encoder file in page cache.
- **The DiT pays instead.** Loading a non-dynamic model makes core move the dynamic DiT off the card, and sampling
  then reloads it.
- **What set the slow encoder pass seen earlier** (about 17 s, in the morning's server log) was most likely that parts
  of the 48 GB encoder file were not in page cache at the time. That attribution is reasoned, not proven by a
  controlled test.
- **Forcing core's layer-ahead prefetch on the conditioning pass changed nothing.**

**Stack.** ComfyUI core at the day's HEAD, flags from `<comfy>/start.sh` default mode (`--fast fp16_accumulation
--cuda-malloc --supports-fp8-compute --mmap-torch-files`, dynamic VRAM on). The encoder file:
- `qwen3vl_32b_minimax_h3_bf16_pruned.safetensors`, 48 GiB on disk, staged by core as 49,118 MB.

The card and machine:
- RTX 4090, its PCIe link at 4.0 x8 (`nvidia-smi`: width 8, max 16).
- 125 GiB RAM, no swap.
- The model store on a PCIe 4.0 x4 NVMe.

## 1. End to end on the server (`:8188`)

**Method.** `keep_in_ram_e2e.py` (session scratchpad; not kept in the repo) submitted the shipped graphs.
- **Per graph and setting:** one warm-up run and two measured runs, each with a sentence appended to the prompt so
  the encoder ran every time.
- **Recorded per run:** submit-to-done wall time, the stage boundaries from the server log ("Requested to load"
  lines) and the server process's peak `VmRSS`.
- **The video arm was** `h3_text_to_video_pika_api.json` at length 141.

| graph | `keep_in_ram` off: warm-up, run 2, run 3 | `keep_in_ram` on: warm-up, run 2, run 3 |
|---|---|---|
| `image/h3_image_t2i_api.json` | 35.4 s, 27.7 s, 27.3 s | 53.0 s, 39.9 s, 40.2 s |
| `image/h3_image_edit_api.json` | 55.0 s, 30.9 s, 30.9 s | 60.5 s, 42.4 s, 44.4 s |
| `h3_text_to_video_pika_api.json` | 86.2 s, 67.7 s, (stopped) | not run |

**Where the time went**, from the log on the warm-up runs. These are the only runs that log every load:

| text to image | encoder | sampling (DiT load included) | decode and shrink |
|---|---|---|---|
| off, warm-up | 7.0 s | 22.6 s | 5.3 s |
| on, run 2 | 6.7 s | 29.3 s | 3.6 s |

**The video "on" arm was stopped before it started.** The harness's `VmRSS` peaked at 115 to 118 GiB on the first
"on" run of each image graph, on a machine with no swap. Checked during the run, the server's own anonymous memory
was about 34 GB, and the rest was file-backed (`RssAnon` against `RssFile`). With the image result already clear, the
video arm was not worth the risk. After the run, `POST /free` brought the server back to 5.3 GB resident.

## 2. In a separate process (not the server)

`encoder_pass_profile.py` (session scratchpad) built the encoder as the server does: core's flags, comfy-aimdo
initialized, `ModelPatcherDynamic`, `comfy.sd.load_clip(CLIPType.MINIMAX)`. It timed one encode of the shipped
text-to-image prompt (326 tokens) per decoder layer, unloading every model between passes.

- **The math runs on the card.** Each layer's input is on `cuda:0`, and the whole machine's CPU sat at 8–10% of 16
  cores during a pass.
- **The hardware is not the limit:**
  - host-to-card copies of 1 GiB: pageable 10.9 GiB/s, pinned 11.2 GiB/s, from the memory-mapped file 10.9 GiB/s;
  - the drive: 6.9 GB/s direct, 9.9 GB/s from page cache (`dd`);
  - core's own reader, `comfy_aimdo.host_buffer.read_file_to_device`, one call per weight: 9.5 GiB/s.
- **Evicted passes, dynamic:** 14.6 to 17.4 s, against the server's 7 s. The forced eviction between passes
  (`unload_all_models`) most likely makes each pass redo setup the server keeps between runs, so these figures
  overstate the server's pass. That is reasoned, not shown.
- **Evicted passes, classic patcher** (`disable_dynamic=True`): 5.7 to 6.8 s. This is the comparison that suggested
  `keep_in_ram`. Section 1 shows it does not survive the DiT coming back.
- **`--disable-fast-disk` was worse:** 16.9 s, then 24.4 and 25.0 s.
- **Forcing prefetch** (`make_prefetch_queue` given `prefetch_dynamic_vbars=True`, which `Llama2_.forward` passes
  only with `past_key_values`): 16.2 to 17.4 s, against 16.3 to 16.9 s without.
- **The conditioning is bitwise identical** across dynamic, classic and forced prefetch, token tags included.

## 3. What this cannot see

- **One prompt, one seed, one machine.** The page-cache account of the 17 s pass rests on one measurement: 7 s with
  the whole file cached (`mincore`) against 17 s earlier in the day, cache state then unknown. It is not a
  controlled test.
- **Peak `VmRSS` counts file-backed pages.** The anonymous peak was not recorded.
- **The video comparison was not run.**
- **The scripts were throwaways in the session's scratchpad.** Their method is described above, so they can be
  written again.
